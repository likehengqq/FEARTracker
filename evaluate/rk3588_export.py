import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from fire import Fire
from hydra.utils import instantiate

from model_training.utils.hydra import load_yaml
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


def _is_state_dict(value: Any) -> bool:
    return isinstance(value, dict) and bool(value) and all(torch.is_tensor(v) for v in value.values())


def _extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint.state_dict()

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Unsupported checkpoint object type '{}'. Expected nn.Module, state_dict, "
            "or a dict containing state_dict/model_state_dict.".format(type(checkpoint).__name__)
        )

    if _is_state_dict(checkpoint):
        return checkpoint

    for key in ("state_dict", "model_state_dict", "model", "net", "module"):
        if key not in checkpoint:
            continue
        value = checkpoint[key]
        if isinstance(value, torch.nn.Module):
            return value.state_dict()
        if _is_state_dict(value):
            return value

    raise KeyError(
        "Could not find model weights in checkpoint. Supported keys are: "
        "state_dict, model_state_dict, model, net, module."
    )


def _strip_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    return {key[len(prefix):] if key.startswith(prefix) else key: value for key, value in state_dict.items()}


def _normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    keys = list(state_dict.keys())

    for prefix in ("module.", "model."):
        if keys and all(key.startswith(prefix) for key in keys):
            state_dict = {key[len(prefix):]: value for key, value in state_dict.items()}
            keys = list(state_dict.keys())

    # PyTorch Lightning checkpoints often store the network under "model.*"
    # together with other module state. Keep the model subtree if it is present.
    model_state = {key[len("model."):]: value for key, value in state_dict.items() if key.startswith("model.")}
    if model_state:
        return model_state

    return _strip_prefix(state_dict, "module.")


def _load_weights(model: torch.nn.Module, weights_path: str, strict: bool) -> torch.nn.Module:
    checkpoint = torch.load(weights_path, map_location="cpu")
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint

    state_dict = _normalize_state_dict_keys(_extract_state_dict(checkpoint))
    try:
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load weights from '{}'. If this is not a FEAR checkpoint, "
            "pass the matching model config with --config_path or retry with "
            "--strict_weights=False for compatible partial weights.".format(weights_path)
        ) from exc
    return model


def _load_model(config_path: str, weights_path: Optional[str], strict_weights: bool) -> torch.nn.Module:
    config = load_yaml(config_path)
    model = instantiate(config)
    if weights_path is not None:
        model = _load_weights(model, weights_path, strict=strict_weights)
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
    strict_weights: bool = True,
) -> None:
    """
    Export FEAR for RK3588 deployment.

    The tracker is exported as two graphs:
    1. template encoder: template image [1,3,128,128] -> template features [1,256,8,8]
    2. tracking head: search image [1,3,256,256] + template features -> bbox/cls maps

    Quantization datasets are RKNN Toolkit2 dataset text files. For the tracking
    graph each line must provide both inputs in Toolkit2's multi-input format.

    Supported weights_path formats:
    - PyTorch Lightning checkpoints with checkpoint["state_dict"]
    - plain torch.save(model.state_dict()) .pt/.pth files
    - checkpoints with model_state_dict/model/net/module keys
    - complete torch.save(model) files
    """
    output_root = Path(output_dir)
    model = _load_model(config_path=config_path, weights_path=weights_path, strict_weights=strict_weights)

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
