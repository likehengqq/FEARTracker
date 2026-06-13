"""Export the FEAR tracker to ONNX for HiSilicon Hi3519 (NNIE) deployment.

The Hi3519 NNIE toolchain (RuyiStudio / nnie_mapper) consumes a Caffe 1.0 or
ONNX model and produces a quantized ``.wk`` file. This script performs the
first step of that pipeline: it splits the FEAR tracker into two NNIE-friendly
sub-graphs and exports each to ONNX.

The tracker runs in two phases (mirroring ``demo_video.py`` / the CoreML
export in ``coreml_convert.py``):

1. **Template branch** -- run once when the target is initialised:
   ``template image [1, 3, 128, 128]  ->  template_features [1, 256, 8, 8]``

2. **Track / search branch** -- run on every frame:
   ``search image [1, 3, 256, 256] + template_features [1, 256, 8, 8]
     ->  bbox [1, 4, 16, 16], cls [1, 1, 16, 16]``

Both branches are pure conv / depthwise-conv / BN / ReLU graphs *except* for
two operators that the Hi3519 NNIE engine does not support natively and that
should be implemented on the ARM/DSP side (see ``docs/hi3519_deployment.md``):

* the cross-correlation ``torch.matmul`` inside ``MobileCorrelation``;
* the final ``torch.exp`` applied to the bbox regression map.

By default these are kept inside the exported graph so the ONNX file is a
faithful copy of the PyTorch model; use ``--split_postprocess`` notes in the
deployment doc if your NNIE SDK version rejects those ops.

Usage::

    PYTHONPATH=. python evaluate/hi3519_convert.py \
        --weights_path=evaluate/checkpoints/FEAR-XS-NoEmbs.ckpt \
        --output_dir=outputs/hi3519 --opset=11 --verify=True
"""
import os
import json
from typing import Optional, Tuple

import numpy as np
import torch
from fire import Fire
from hydra.utils import instantiate

from model_training.utils.hydra import load_yaml
from model_training.utils.torch import load_from_lighting

# NNIE mapper expects the normalisation to be configured in its prototxt
# (data_scale / mean file). These match the training-time preprocessing in
# model_training/tracker/base_tracker.py and the CoreML export.
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

TEMPLATE_SHAPE = [1, 3, 128, 128]
SEARCH_SHAPE = [1, 3, 256, 256]
TEMPLATE_FEATURES_SHAPE = [1, 256, 8, 8]


class TemplateBranch(torch.nn.Module):
    """Initialisation graph: template image -> template features."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, template: torch.Tensor) -> torch.Tensor:
        return self.model.get_features(template)


class TrackBranch(torch.nn.Module):
    """Per-frame graph: search image + template features -> bbox, cls."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, search: torch.Tensor, template_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pred = self.model.track(search, template_features)
        return pred["TARGET_REGRESSION_LABEL_KEY"], pred["TARGET_CLASSIFICATION_KEY"]


def _export(module: torch.nn.Module, args: Tuple[torch.Tensor, ...], input_names, output_names, path: str, opset: int):
    torch.onnx.export(
        module,
        args,
        path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        # Hi3519 NNIE only supports fixed input shapes, so do NOT mark any axis
        # as dynamic. Batch size is pinned to 1.
        dynamic_axes=None,
    )
    print(f"[ok] exported {path}")


def _verify(module: torch.nn.Module, args: Tuple[torch.Tensor, ...], output_names, path: str) -> None:
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("[skip] onnx / onnxruntime not installed, skipping numerical verification")
        return

    onnx.checker.check_model(onnx.load(path))
    with torch.no_grad():
        torch_out = module(*args)
    if isinstance(torch_out, torch.Tensor):
        torch_out = (torch_out,)

    sess = ort.InferenceSession(path)
    feed = {i.name: a.cpu().numpy() for i, a in zip(sess.get_inputs(), args)}
    ort_out = sess.run(None, feed)

    for name, t, o in zip(output_names, torch_out, ort_out):
        diff = float(np.abs(t.cpu().numpy() - o).max())
        status = "ok" if diff < 1e-3 else "WARN"
        print(f"[{status}] {os.path.basename(path)}::{name} max|torch-onnx| = {diff:.2e}")


def main(
    config_path: str = "model_training/config/model/fear.yaml",
    weights_path: Optional[str] = "evaluate/checkpoints/FEAR-XS-NoEmbs.ckpt",
    output_dir: str = "outputs/hi3519",
    opset: int = 11,
    verify: bool = True,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    config = load_yaml(config_path)
    model = instantiate(config)
    if weights_path is not None:
        model = load_from_lighting(model, weights_path, map_location="cpu")
    model = model.eval()

    template_branch = TemplateBranch(model).eval()
    track_branch = TrackBranch(model).eval()

    template_inp = (torch.rand(*TEMPLATE_SHAPE),)
    track_inp = (torch.rand(*SEARCH_SHAPE), torch.rand(*TEMPLATE_FEATURES_SHAPE))

    template_path = os.path.join(output_dir, "fear_template.onnx")
    track_path = os.path.join(output_dir, "fear_track.onnx")

    _export(template_branch, template_inp, ["template"], ["template_features"], template_path, opset)
    _export(track_branch, track_inp, ["search", "template_features"], ["bbox", "cls"], track_path, opset)

    if verify:
        _verify(template_branch, template_inp, ["template_features"], template_path)
        _verify(track_branch, track_inp, ["bbox", "cls"], track_path)

    # Preprocessing / IO description that the NNIE mapper prototxt must match.
    meta = {
        "opset": opset,
        "preprocess": {
            "color": "RGB",
            "mean_0_1": _MEAN,
            "std_0_1": _STD,
            "comment": "pixel/255 then (x-mean)/std; for NNIE set img_norm + data_scale accordingly",
        },
        "graphs": {
            "fear_template.onnx": {
                "inputs": {"template": TEMPLATE_SHAPE},
                "outputs": {"template_features": TEMPLATE_FEATURES_SHAPE},
                "run": "once on target initialisation",
            },
            "fear_track.onnx": {
                "inputs": {"search": SEARCH_SHAPE, "template_features": TEMPLATE_FEATURES_SHAPE},
                "outputs": {"bbox": [1, 4, 16, 16], "cls": [1, 1, 16, 16]},
                "run": "every frame",
            },
        },
        "unsupported_by_nnie": [
            "MobileCorrelation matmul (cross-correlation) -- run on CPU/DSP",
            "torch.exp on bbox regression map -- run on CPU/DSP",
        ],
    }
    meta_path = os.path.join(output_dir, "fear_hi3519_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[ok] wrote {meta_path}")


if __name__ == "__main__":
    Fire(main)
