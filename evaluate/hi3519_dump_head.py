"""导出 Hi3519 CPU 头部段（BoxTower）的权重，供 C++ 板端实现加载。

NNIE 跑不了头部的互相关 MatMul / Exp，头部需在 ARM 上用 C++ 实现（见 evaluate/hi3519_cpp）。
本脚本把 BoxTower 的权重**折叠 BatchNorm 到逐点卷积**后，按固定顺序写成一个 float32 二进制文件
``hi3519_head_weights.bin``，C++ 端按相同顺序读取。

同时（可选）dump 一组参考数据 ``hi3519_head_ref.bin``（随机 search_features / template_features
及对应 PyTorch 输出 bbox / cls），用于 C++ 头部的数值自检。

权重顺序（每个卷积块依次写 depthwise 权重[C,3,3]、depthwise bias[C]、pointwise 权重[Cout,Cin]、
pointwise bias[Cout]）：
  1 cls_encode      C=256  Cout=256 Cin=256  (relu)
  2 reg_encode      C=256  Cout=256 Cin=256  (relu)
  3 cls_dw.enc      C=320  Cout=256 Cin=320  (relu)
  4 reg_dw.enc      C=320  Cout=256 Cin=320  (relu)
  5 bbox_tower[0]   C=256  Cout=256 Cin=256  (relu)
  6 bbox_tower[3]   C=256  Cout=256 Cin=256  (relu)
  7 cls_tower[0]    C=256  Cout=256 Cin=256  (relu)
  8 cls_tower[3]    C=256  Cout=256 Cin=256  (relu)
  9 bbox_pred       C=256  Cout=4   Cin=256  (无 BN/relu)
 10 cls_pred        C=256  Cout=1   Cin=256  (无 BN/relu)
最后写 adjust(1) 与 bias(4)。

用法::

    PYTHONPATH=. python evaluate/hi3519_dump_head.py --output_dir=outputs/hi3519_split
"""
import os
from typing import Optional, Tuple

import numpy as np
import torch
from fire import Fire
from hydra.utils import instantiate

from model_training.utils.hydra import load_yaml
from model_training.utils.torch import load_from_lighting

_EPS = 1e-5


def _fold(sepconv, bn) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """把 SepConv(+可选 BN) 折叠成 depthwise 权重/bias 与（折叠 BN 后的）pointwise 权重/bias。"""
    dw_w = sepconv.depthwise.weight.detach().cpu().numpy()  # [C,1,3,3]
    C = dw_w.shape[0]
    dw_w = dw_w.reshape(C, 3, 3)
    dw_b = (
        sepconv.depthwise.bias.detach().cpu().numpy()
        if sepconv.depthwise.bias is not None else np.zeros(C, np.float32)
    )
    pw_w = sepconv.pointwise.weight.detach().cpu().numpy()[:, :, 0, 0]  # [Cout,Cin]
    pw_b = (
        sepconv.pointwise.bias.detach().cpu().numpy()
        if sepconv.pointwise.bias is not None else np.zeros(pw_w.shape[0], np.float32)
    )
    if bn is not None:
        gamma = bn.weight.detach().cpu().numpy()
        beta = bn.bias.detach().cpu().numpy()
        mean = bn.running_mean.detach().cpu().numpy()
        var = bn.running_var.detach().cpu().numpy()
        s = gamma / np.sqrt(var + _EPS)
        pw_w = pw_w * s[:, None]
        pw_b = pw_b * s + (beta - mean * s)
    return (
        dw_w.astype(np.float32), dw_b.astype(np.float32),
        pw_w.astype(np.float32), pw_b.astype(np.float32),
    )


def _write_block(f, sepconv, bn) -> None:
    for arr in _fold(sepconv, bn):
        f.write(np.ascontiguousarray(arr, dtype="<f4").tobytes())


def main(
    config_path: str = "model_training/config/model/fear.yaml",
    weights_path: Optional[str] = "evaluate/checkpoints/FEAR-XS-NoEmbs.ckpt",
    output_dir: str = "outputs/hi3519_split",
    dump_reference: bool = True,
    seed: int = 0,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    model = instantiate(load_yaml(config_path))
    if weights_path is not None:
        model = load_from_lighting(model, weights_path, map_location="cpu")
    model = model.eval()
    bt = model.connect_model

    weights_path_out = os.path.join(output_dir, "hi3519_head_weights.bin")
    with open(weights_path_out, "wb") as f:
        _write_block(f, bt.cls_encode.matrix11_s[0], bt.cls_encode.matrix11_s[1])
        _write_block(f, bt.reg_encode.matrix11_s[0], bt.reg_encode.matrix11_s[1])
        _write_block(f, bt.cls_dw.enc[0], bt.cls_dw.enc[1])
        _write_block(f, bt.reg_dw.enc[0], bt.reg_dw.enc[1])
        _write_block(f, bt.bbox_tower[0], bt.bbox_tower[1])
        _write_block(f, bt.bbox_tower[3], bt.bbox_tower[4])
        _write_block(f, bt.cls_tower[0], bt.cls_tower[1])
        _write_block(f, bt.cls_tower[3], bt.cls_tower[4])
        _write_block(f, bt.bbox_pred, None)
        _write_block(f, bt.cls_pred, None)
        adjust = bt.adjust.detach().cpu().numpy().astype("<f4")
        bias = bt.bias.detach().cpu().numpy().reshape(-1).astype("<f4")
        f.write(np.ascontiguousarray(adjust).tobytes())
        f.write(np.ascontiguousarray(bias).tobytes())
    print("[ok] 已写入头部权重 {} ({} 字节)".format(weights_path_out, os.path.getsize(weights_path_out)))

    if dump_reference:
        torch.manual_seed(seed)
        sf = torch.rand(1, 256, 16, 16)
        tf = torch.rand(1, 256, 8, 8)
        with torch.no_grad():
            pred = model.connector(template_features=tf, search_features=sf)
            bbox = pred["TARGET_REGRESSION_LABEL_KEY"].cpu().numpy()
            cls = pred["TARGET_CLASSIFICATION_KEY"].cpu().numpy()
        ref_path = os.path.join(output_dir, "hi3519_head_ref.bin")
        with open(ref_path, "wb") as f:
            for arr in (sf.numpy(), tf.numpy(), bbox, cls):
                f.write(np.ascontiguousarray(arr, dtype="<f4").tobytes())
        print("[ok] 已写入参考数据 {} (search_features/template_features/bbox/cls)".format(ref_path))


if __name__ == "__main__":
    Fire(main)
