"""转 .rknn 前后的数值/精度对齐脚本。

对 FEAR 的两个子图（模板分支、跟踪分支），用同一组输入分别跑：

- PyTorch（参考基准）
- ONNX（onnxruntime）
- RKNN（rknn-toolkit2 模拟器，可选；含可选 INT8 量化）

并逐输出报告 **最大绝对误差** 与 **余弦相似度**，用于判断导出与量化是否引入明显误差。

PyTorch 与 ONNX 的对比在任意 x86 主机即可运行；RKNN 一栏需要安装 ``rknn-toolkit2``，
否则会自动跳过。

用法（仅 PyTorch vs ONNX，自检）::

    PYTHONPATH=. python evaluate/rk3588_accuracy.py \
        --template_onnx=outputs/rv1126b/fear_template_encoder.onnx \
        --track_onnx=outputs/rv1126b/fear_track.onnx

加上 RKNN 模拟器与 INT8 量化对比::

    PYTHONPATH=. python evaluate/rk3588_accuracy.py \
        --target_platform=rv1126b --check_rknn=True \
        --do_quantization=True \
        --template_dataset=/path/to/template_dataset.txt \
        --track_dataset=/path/to/track_dataset.txt
"""
from typing import List, Optional, Sequence

import numpy as np
import torch
from fire import Fire
from hydra.utils import instantiate

from model_training.utils.hydra import load_yaml
from model_training.utils.torch import load_from_lighting
from evaluate.rk3588_export import TemplateFeatureWrapper, RKNNTrackingWrapper

DEFAULT_CONFIG_PATH = "model_training/config/model/fear.yaml"
DEFAULT_WEIGHTS_PATH = "evaluate/checkpoints/FEAR-XS-NoEmbs.ckpt"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def _report(name: str, ref: np.ndarray, other: np.ndarray) -> None:
    max_abs = float(np.abs(ref.astype(np.float64) - other.astype(np.float64)).max())
    cos = _cosine(ref, other)
    flag = "ok" if (cos > 0.999 and max_abs < 1e-2) else "CHECK"
    print("  [{}] {:<28} 最大绝对误差={:.3e}  余弦相似度={:.6f}".format(flag, name, max_abs, cos))


def _onnx_infer(onnx_path: str, inputs: Sequence[np.ndarray]) -> List[np.ndarray]:
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path)
    feed = {i.name: np.asarray(a, dtype=np.float32) for i, a in zip(sess.get_inputs(), inputs)}
    return sess.run(None, feed)


def _rknn_infer(
    onnx_path: str,
    inputs: Sequence[np.ndarray],
    target_platform: str,
    do_quantization: bool,
    dataset: Optional[str],
) -> Optional[List[np.ndarray]]:
    try:
        from rknn.api import RKNN
    except ImportError:
        print("  [skip] 未安装 rknn-toolkit2，跳过 RKNN 模拟器对比")
        return None

    rknn = RKNN(verbose=False)
    try:
        rknn.config(target_platform=target_platform)
        if rknn.load_onnx(model=onnx_path) != 0:
            raise RuntimeError("load_onnx 失败: {}".format(onnx_path))
        if rknn.build(do_quantization=do_quantization, dataset=dataset) != 0:
            raise RuntimeError("build 失败: {}".format(onnx_path))
        if rknn.init_runtime(target=None) != 0:  # target=None -> PC 模拟器
            raise RuntimeError("init_runtime(模拟器) 失败: {}".format(onnx_path))
        return rknn.inference(inputs=[np.asarray(a) for a in inputs])
    finally:
        rknn.release()


def main(
    config_path: str = DEFAULT_CONFIG_PATH,
    weights_path: Optional[str] = DEFAULT_WEIGHTS_PATH,
    template_onnx: str = "outputs/rv1126b/fear_template_encoder.onnx",
    track_onnx: str = "outputs/rv1126b/fear_track.onnx",
    check_rknn: bool = False,
    target_platform: str = "rv1126b",
    do_quantization: bool = False,
    template_dataset: Optional[str] = None,
    track_dataset: Optional[str] = None,
    seed: int = 0,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = instantiate(load_yaml(config_path))
    if weights_path is not None:
        model = load_from_lighting(model, weights_path, map_location="cpu")
    model = model.eval()

    template_in = torch.rand(1, 3, 128, 128)
    search_in = torch.rand(1, 3, 256, 256)
    tmpl_feat_in = torch.rand(1, 256, 8, 8)

    with torch.no_grad():
        tmpl_pt = TemplateFeatureWrapper(model).eval()(template_in).cpu().numpy()
        bbox_pt, cls_pt = RKNNTrackingWrapper(model).eval()(search_in, tmpl_feat_in)
        bbox_pt, cls_pt = bbox_pt.cpu().numpy(), cls_pt.cpu().numpy()

    print("================ 模板分支 (template) ================")
    tmpl_onnx = _onnx_infer(template_onnx, [template_in.numpy()])[0]
    print(" PyTorch vs ONNX:")
    _report("template_features", tmpl_pt, tmpl_onnx)
    if check_rknn:
        tmpl_rknn = _rknn_infer(template_onnx, [template_in.numpy()], target_platform, do_quantization, template_dataset)
        if tmpl_rknn is not None:
            print(" PyTorch vs RKNN{}:".format("(INT8)" if do_quantization else ""))
            _report("template_features", tmpl_pt, np.asarray(tmpl_rknn[0]).reshape(tmpl_pt.shape))

    print("================ 跟踪分支 (track) ===================")
    onnx_out = _onnx_infer(track_onnx, [search_in.numpy(), tmpl_feat_in.numpy()])
    print(" PyTorch vs ONNX:")
    _report("bbox", bbox_pt, onnx_out[0])
    _report("cls", cls_pt, onnx_out[1])
    if check_rknn:
        rknn_out = _rknn_infer(
            track_onnx, [search_in.numpy(), tmpl_feat_in.numpy()], target_platform, do_quantization, track_dataset
        )
        if rknn_out is not None:
            print(" PyTorch vs RKNN{}:".format("(INT8)" if do_quantization else ""))
            _report("bbox", bbox_pt, np.asarray(rknn_out[0]).reshape(bbox_pt.shape))
            _report("cls", cls_pt, np.asarray(rknn_out[1]).reshape(cls_pt.shape))
    print("====================================================")


if __name__ == "__main__":
    Fire(main)
