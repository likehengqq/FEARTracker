"""将 FEAR 跟踪器导出为 ONNX，用于海思 Hi3519（NNIE）部署。

Hi3519 的 NNIE 工具链（RuyiStudio / nnie_mapper）接受 Caffe 1.0 或 ONNX 模型，
并生成量化后的 ``.wk`` 文件。本脚本完成该链路的第一步：把 FEAR 跟踪器拆成两个
对 NNIE 友好的子图，并分别导出为 ONNX。

跟踪器分两个阶段运行（与 ``demo_video.py`` / ``coreml_convert.py`` 的 CoreML 导出一致）：

1. **模板分支** —— 仅在跟踪初始化时运行一次：
   ``模板图像 [1, 3, 128, 128]  ->  模板特征 template_features [1, 256, 8, 8]``

2. **跟踪/搜索分支** —— 每帧运行：
   ``搜索图像 search [1, 3, 256, 256] + 模板特征 template_features [1, 256, 8, 8]
     ->  bbox [1, 4, 16, 16], cls [1, 1, 16, 16]``

除以下两个算子外，两个分支都是纯 conv / depthwise-conv / BN / ReLU 计算图。
这两个算子 Hi3519 NNIE 引擎不原生支持，应放到 ARM/DSP 端实现
（详见 ``docs/hi3519_deployment.md``）：

* ``MobileCorrelation`` 内部的互相关 ``torch.matmul``；
* 对 bbox 回归图施加的最后那个 ``torch.exp``。

默认情况下这两个算子仍保留在导出的计算图中，使 ONNX 文件与 PyTorch 模型完全一致；
若你的 NNIE SDK 版本拒绝这些算子，请参考部署文档中的网络拆分策略。

用法::

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

# NNIE mapper 需要在其 prototxt（data_scale / mean 文件）中配置归一化参数。
# 下列取值与 model_training/tracker/base_tracker.py 中训练时的预处理以及 CoreML 导出一致。
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

TEMPLATE_SHAPE = [1, 3, 128, 128]
SEARCH_SHAPE = [1, 3, 256, 256]
TEMPLATE_FEATURES_SHAPE = [1, 256, 8, 8]


class TemplateBranch(torch.nn.Module):
    """初始化计算图：模板图像 -> 模板特征。"""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, template: torch.Tensor) -> torch.Tensor:
        return self.model.get_features(template)


class TrackBranch(torch.nn.Module):
    """每帧计算图：搜索图像 + 模板特征 -> bbox, cls。"""

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
        # Hi3519 NNIE 仅支持固定输入尺寸，因此不要把任何维度标记为动态。batch 固定为 1。
        dynamic_axes=None,
    )
    print(f"[ok] 已导出 {path}")


def _verify(module: torch.nn.Module, args: Tuple[torch.Tensor, ...], output_names, path: str) -> None:
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("[skip] 未安装 onnx / onnxruntime，跳过数值校验")
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
        print(f"[{status}] {os.path.basename(path)}::{name} 最大绝对误差|torch-onnx| = {diff:.2e}")


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

    # NNIE mapper 的 prototxt 必须与下列预处理 / 输入输出描述保持一致。
    meta = {
        "opset": opset,
        "preprocess": {
            "color": "RGB",
            "mean_0_1": _MEAN,
            "std_0_1": _STD,
            "comment": "先 pixel/255，再 (x-mean)/std；NNIE 中需相应配置 img_norm + data_scale",
        },
        "graphs": {
            "fear_template.onnx": {
                "inputs": {"template": TEMPLATE_SHAPE},
                "outputs": {"template_features": TEMPLATE_FEATURES_SHAPE},
                "run": "跟踪初始化时运行一次",
            },
            "fear_track.onnx": {
                "inputs": {"search": SEARCH_SHAPE, "template_features": TEMPLATE_FEATURES_SHAPE},
                "outputs": {"bbox": [1, 4, 16, 16], "cls": [1, 1, 16, 16]},
                "run": "每帧运行",
            },
        },
        "unsupported_by_nnie": [
            "MobileCorrelation 的 matmul（互相关）—— 放到 CPU/DSP 执行",
            "bbox 回归图上的 torch.exp —— 放到 CPU/DSP 执行",
        ],
    }
    meta_path = os.path.join(output_dir, "fear_hi3519_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[ok] 已写入 {meta_path}")


if __name__ == "__main__":
    Fire(main)
