import os
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
from fire import Fire
from hydra.utils import instantiate

from model_training.utils.hydra import load_yaml
from model_training.utils.torch import load_from_lighting
from model_training.utils.constants import (
    TARGET_CLASSIFICATION_KEY,
    TARGET_REGRESSION_LABEL_KEY,
)


DEFAULT_CONFIG_PATH = "model_training/config/model/fear.yaml"
DEFAULT_WEIGHTS_PATH = "evaluate/checkpoints/FEAR-XS-NoEmbs.ckpt"
DEFAULT_OUTPUT_DIR = "outputs/rk3588"


class TemplateFeatureWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, template: torch.Tensor) -> torch.Tensor:
        return self.model.get_features(template)


class RKNNTrackingWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, search: torch.Tensor, template_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pred = self.model.track(search, template_features)
        return pred[TARGET_REGRESSION_LABEL_KEY], pred[TARGET_CLASSIFICATION_KEY]


def _load_model(config_path: str, weights_path: Optional[str]) -> torch.nn.Module:
    config = load_yaml(config_path)
    model = instantiate(config)
    if weights_path is not None:
        model = load_from_lighting(model, weights_path, map_location="cpu")
    return model.eval()


def _export_onnx(
    model: torch.nn.Module,
    inputs: Sequence[torch.Tensor],
    output_path: Path,
    input_names: Sequence[str],
    output_names: Sequence[str],
    opset_version: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model,
            tuple(inputs),
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=list(input_names),
            output_names=list(output_names),
        )


def _build_rknn(
    onnx_path: Path,
    rknn_path: Path,
    target_platform: str,
    do_quantization: bool,
    dataset: Optional[str],
) -> None:
    try:
        from rknn.api import RKNN
    except ImportError as exc:
        raise ImportError(
            "rknn-toolkit2 is required to build .rknn files. Install the Rockchip "
            "wheel on the export host, or rerun with skip_rknn=True to only export ONNX."
        ) from exc

    if do_quantization and dataset is None:
        raise ValueError("A calibration dataset file is required when do_quantization=True.")

    rknn = RKNN(verbose=True)
    try:
        ret = rknn.config(target_platform=target_platform)
        if ret != 0:
            raise RuntimeError("RKNN config failed for {} with code {}".format(target_platform, ret))

        ret = rknn.load_onnx(model=str(onnx_path))
        if ret != 0:
            raise RuntimeError("RKNN load_onnx failed for {} with code {}".format(onnx_path, ret))

        ret = rknn.build(do_quantization=do_quantization, dataset=dataset)
        if ret != 0:
            raise RuntimeError("RKNN build failed for {} with code {}".format(onnx_path, ret))

        rknn_path.parent.mkdir(parents=True, exist_ok=True)
        ret = rknn.export_rknn(str(rknn_path))
        if ret != 0:
            raise RuntimeError("RKNN export failed for {} with code {}".format(rknn_path, ret))
    finally:
        rknn.release()


def main(
    config_path: str = DEFAULT_CONFIG_PATH,
    weights_path: Optional[str] = DEFAULT_WEIGHTS_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    target_platform: str = "rk3588",
    opset_version: int = 12,
    skip_rknn: bool = False,
    do_quantization: bool = False,
    template_dataset: Optional[str] = None,
    track_dataset: Optional[str] = None,
) -> None:
    """
    Export FEAR for RK3588 deployment.

    The tracker is exported as two graphs:
    1. template encoder: template image [1,3,128,128] -> template features [1,256,8,8]
    2. tracking head: search image [1,3,256,256] + template features -> bbox/cls maps

    Quantization datasets are RKNN Toolkit2 dataset text files. For the tracking
    graph each line must provide both inputs in Toolkit2's multi-input format.
    """
    output_root = Path(output_dir)
    model = _load_model(config_path=config_path, weights_path=weights_path)

    template_input = torch.randn(1, 3, 128, 128)
    search_input = torch.randn(1, 3, 256, 256)
    template_features = torch.randn(1, 256, 8, 8)

    template_onnx = output_root / "fear_template_encoder.onnx"
    track_onnx = output_root / "fear_track.onnx"
    template_rknn = output_root / "fear_template_encoder.rknn"
    track_rknn = output_root / "fear_track.rknn"

    _export_onnx(
        TemplateFeatureWrapper(model).eval(),
        [template_input],
        template_onnx,
        input_names=["template"],
        output_names=["template_features"],
        opset_version=opset_version,
    )
    _export_onnx(
        RKNNTrackingWrapper(model).eval(),
        [search_input, template_features],
        track_onnx,
        input_names=["search", "template_features"],
        output_names=["bbox", "cls"],
        opset_version=opset_version,
    )

    if skip_rknn:
        print("Exported ONNX models to {}".format(output_root))
        return

    _build_rknn(
        onnx_path=template_onnx,
        rknn_path=template_rknn,
        target_platform=target_platform,
        do_quantization=do_quantization,
        dataset=template_dataset,
    )
    _build_rknn(
        onnx_path=track_onnx,
        rknn_path=track_rknn,
        target_platform=target_platform,
        do_quantization=do_quantization,
        dataset=track_dataset,
    )
    print("Exported RKNN models to {}".format(os.fspath(output_root)))


if __name__ == "__main__":
    Fire(main)
