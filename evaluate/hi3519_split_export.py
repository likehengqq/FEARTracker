"""为 Hi3519（NNIE）板端部署做「NNIE 骨干 + CPU 头部」的三段式导出。

Hi3519 的 NNIE 引擎不支持互相关 ``MatMul``、``Exp`` 以及动态 reshape，因此本脚本把
FEAR 跟踪器切成两类计算图：

- **NNIE 段（纯卷积，计算量主要在这）**：骨干特征提取 ``get_features``（encoder + neck）
  - ``hi3519_backbone_template.onnx``：模板图像 [1,3,128,128] -> 模板特征 [1,256,8,8]（初始化跑一次）
  - ``hi3519_backbone_search.onnx``：搜索图像 [1,3,256,256] -> 搜索特征 [1,256,16,16]（每帧）
- **CPU 段（含 MatMul/Exp，作用在 16×16 小特征图上，开销小）**：BoxTower 头部
  - ``hi3519_head.onnx``：(搜索特征 [1,256,16,16], 模板特征 [1,256,8,8]) -> bbox [1,4,16,16], cls [1,1,16,16]

该切分在数值上与原始 ``track()`` 完全等价（误差 0）。NNIE 段交给 nnie_mapper 转成 ``.wk``；
CPU 段在板端用 ARM 实现（参考 ``evaluate/hi3519_runtime.py`` 的 numpy/onnx 版本，或
``evaluate/hi3519_cpp`` 的 C++ 版本）。

用法::

    PYTHONPATH=. python evaluate/hi3519_split_export.py --output_dir=outputs/hi3519_split
"""
import os
import json
from typing import Optional, Tuple

import numpy as np
import torch
from fire import Fire
from hydra.utils import instantiate

from model_training.utils.hydra import load_hydra_config_from_path, load_yaml
from model_training.utils.torch import load_from_lighting

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

TEMPLATE_SHAPE = [1, 3, 128, 128]
SEARCH_SHAPE = [1, 3, 256, 256]
TEMPLATE_FEATURES_SHAPE = [1, 256, 8, 8]
SEARCH_FEATURES_SHAPE = [1, 256, 16, 16]


class BackboneBranch(torch.nn.Module):
    """NNIE 段：图像 -> 骨干特征（encoder + neck）。"""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model.get_features(image)


class HeadBranch(torch.nn.Module):
    """CPU 段：搜索特征 + 模板特征 -> bbox, cls（含互相关 MatMul 与 Exp）。"""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, search_features: torch.Tensor, template_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pred = self.model.connector(template_features=template_features, search_features=search_features)
        return pred["TARGET_REGRESSION_LABEL_KEY"], pred["TARGET_CLASSIFICATION_KEY"]


def _export(module, args, input_names, output_names, path, opset):
    torch.onnx.export(
        module, args, path, export_params=True, opset_version=opset,
        do_constant_folding=True, input_names=input_names, output_names=output_names,
        dynamic_axes=None,
    )
    print(f"[ok] 已导出 {path}")


def _verify(module, args, output_names, path):
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("[skip] 未安装 onnx / onnxruntime，跳过数值校验")
        return
    onnx.checker.check_model(onnx.load(path))
    with torch.no_grad():
        ref = module(*args)
    if isinstance(ref, torch.Tensor):
        ref = (ref,)
    sess = ort.InferenceSession(path)
    feed = {i.name: a.cpu().numpy() for i, a in zip(sess.get_inputs(), args)}
    out = sess.run(None, feed)
    for name, t, o in zip(output_names, ref, out):
        diff = float(np.abs(t.cpu().numpy() - o).max())
        status = "ok" if diff < 1e-3 else "WARN"
        print(f"[{status}] {os.path.basename(path)}::{name} 最大绝对误差 = {diff:.2e}")


def main(
    config_path: str = "model_training/config/model/fear.yaml",
    weights_path: Optional[str] = "evaluate/checkpoints/FEAR-XS-NoEmbs.ckpt",
    output_dir: str = "outputs/hi3519_split",
    opset: int = 11,
    verify: bool = True,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    model = instantiate(load_yaml(config_path))
    if weights_path is not None:
        model = load_from_lighting(model, weights_path, map_location="cpu")
    model = model.eval()

    backbone = BackboneBranch(model).eval()
    head = HeadBranch(model).eval()

    tmpl_in = (torch.rand(*TEMPLATE_SHAPE),)
    search_in = (torch.rand(*SEARCH_SHAPE),)
    head_in = (torch.rand(*SEARCH_FEATURES_SHAPE), torch.rand(*TEMPLATE_FEATURES_SHAPE))

    tmpl_path = os.path.join(output_dir, "hi3519_backbone_template.onnx")
    search_path = os.path.join(output_dir, "hi3519_backbone_search.onnx")
    head_path = os.path.join(output_dir, "hi3519_head.onnx")

    _export(backbone, tmpl_in, ["template"], ["template_features"], tmpl_path, opset)
    _export(backbone, search_in, ["search"], ["search_features"], search_path, opset)
    _export(head, head_in, ["search_features", "template_features"], ["bbox", "cls"], head_path, opset)

    if verify:
        _verify(backbone, tmpl_in, ["template_features"], tmpl_path)
        _verify(backbone, search_in, ["search_features"], search_path)
        _verify(head, head_in, ["bbox", "cls"], head_path)

    meta = {
        "opset": opset,
        "preprocess": {"color": "RGB", "mean_0_1": _MEAN, "std_0_1": _STD,
                       "comment": "先 pixel/255，再 (x-mean)/std；NNIE 段在 nnie_mapper 里配置归一化"},
        "nnie_graphs": {
            "hi3519_backbone_template.onnx": {"inputs": {"template": TEMPLATE_SHAPE},
                                              "outputs": {"template_features": TEMPLATE_FEATURES_SHAPE},
                                              "run": "初始化跑一次"},
            "hi3519_backbone_search.onnx": {"inputs": {"search": SEARCH_SHAPE},
                                            "outputs": {"search_features": SEARCH_FEATURES_SHAPE},
                                            "run": "每帧"},
        },
        "cpu_graph": {
            "hi3519_head.onnx": {"inputs": {"search_features": SEARCH_FEATURES_SHAPE,
                                            "template_features": TEMPLATE_FEATURES_SHAPE},
                                 "outputs": {"bbox": [1, 4, 16, 16], "cls": [1, 1, 16, 16]},
                                 "run": "每帧，在 ARM CPU 上（含 MatMul/Exp）"},
        },
        "note": "NNIE 段（两个 backbone）转 .wk 上 NNIE；CPU 段（head）在 ARM 上实现。",
    }
    with open(os.path.join(output_dir, "hi3519_split_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[ok] 已写入 {os.path.join(output_dir, 'hi3519_split_meta.json')}")


if __name__ == "__main__":
    Fire(main)
