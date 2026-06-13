"""Hi3519（NNIE）分段性能基准。

按 FEAR 真实推理流程逐帧运行 NNIE 骨干 + CPU 头部的参考实现，并把每帧耗时拆成：

- 预处理（CPU）：裁剪后归一化、转 NCHW
- NNIE 骨干（搜索）：搜索分支 backbone 推理（板端在 NNIE 上）
- CPU 头部：BoxTower（互相关 MatMul + Exp）
- 解码（CPU）：sigmoid / argmax / 网格解码 / 还原
- 其它（CPU）：搜索区域裁剪等

并单独报告模板骨干（初始化跑一次）的耗时，输出端到端 FPS。

默认 backend=onnx（onnxruntime 模拟三段）。**注意：onnx 后端下“NNIE 骨干”一栏是 CPU 上
onnxruntime 的耗时，不代表 Hi3519 NNIE 的真实性能**；真实分段耗时需在板端用 C++/NNIE 实现测量。

用法::

    PYTHONPATH=. python evaluate/hi3519_benchmark.py --output_dir=outputs/hi3519_split \
        --video_path=assets/test.mp4 --num_frames=200
"""
import os
import time
from typing import List, Optional

import numpy as np
from fire import Fire


def _load_frames(video_path: Optional[str], num_frames: int, synth_size) -> List[np.ndarray]:
    if video_path:
        import imageio.v3 as iio

        video = iio.imread(video_path)
        frames = [np.asarray(f) for f in video]
        if not frames:
            raise RuntimeError("视频没有帧: {}".format(video_path))
        out = []
        i = 0
        while len(out) < num_frames + 1:
            out.append(frames[i % len(frames)])
            i += 1
        return out
    h, w = synth_size
    rng = np.random.default_rng(0)
    return [rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8) for _ in range(num_frames + 1)]


class _SegmentTimer:
    """包裹运行时方法，分段累计耗时（不修改 hi3519_runtime.py）。"""

    def __init__(self, tracker) -> None:
        self.acc = {"preprocess": 0.0, "nnie": 0.0, "head": 0.0, "decode": 0.0, "total": 0.0}
        self._orig_pre = tracker._preprocess_image
        self._orig_post = tracker._postprocess
        self._orig_search = tracker.backend.backbone_search
        self._orig_head = tracker.backend.head

        def timed_pre(image):
            t = time.perf_counter(); r = self._orig_pre(image); self.acc["preprocess"] += time.perf_counter() - t; return r

        def timed_search(x):
            t = time.perf_counter(); r = self._orig_search(x); self.acc["nnie"] += time.perf_counter() - t; return r

        def timed_head(sf, tf):
            t = time.perf_counter(); r = self._orig_head(sf, tf); self.acc["head"] += time.perf_counter() - t; return r

        def timed_post(*a, **k):
            t = time.perf_counter(); r = self._orig_post(*a, **k); self.acc["decode"] += time.perf_counter() - t; return r

        tracker._preprocess_image = timed_pre
        tracker.backend.backbone_search = timed_search
        tracker.backend.head = timed_head
        tracker._postprocess = timed_post

    def reset(self):
        for k in self.acc:
            self.acc[k] = 0.0


def main(
    output_dir: str = "outputs/hi3519_split",
    video_path: Optional[str] = "assets/test.mp4",
    initial_bbox: List[int] = [163, 53, 45, 174],
    num_frames: int = 200,
    warmup: int = 10,
    backend: str = "onnx",
    synth_height: int = 720,
    synth_width: int = 1280,
) -> None:
    from evaluate.hi3519_runtime import Hi3519FEARTracker

    tracker = Hi3519FEARTracker(
        template_backbone_path=os.path.join(output_dir, "hi3519_backbone_template.onnx"),
        search_backbone_path=os.path.join(output_dir, "hi3519_backbone_search.onnx"),
        head_path=os.path.join(output_dir, "hi3519_head.onnx"),
        backend=backend,
    )

    frames = _load_frames(video_path, num_frames, (synth_height, synth_width))
    init_bbox = np.array(initial_bbox, dtype=np.int32)

    t0 = time.perf_counter()
    tracker.initialize(frames[0], init_bbox)
    template_ms = (time.perf_counter() - t0) * 1000.0

    timer = _SegmentTimer(tracker)
    for i in range(1, min(warmup, len(frames) - 1) + 1):
        tracker.update(frames[i])
    timer.reset()

    n = 0
    loop_start = time.perf_counter()
    for i in range(1, len(frames)):
        ft = time.perf_counter()
        tracker.update(frames[i])
        timer.acc["total"] += time.perf_counter() - ft
        n += 1
        if n >= num_frames:
            break
    wall = time.perf_counter() - loop_start

    acc = timer.acc
    per_frame_ms = (acc["total"] / n) * 1000.0
    pre_ms = (acc["preprocess"] / n) * 1000.0
    nnie_ms = (acc["nnie"] / n) * 1000.0
    head_ms = (acc["head"] / n) * 1000.0
    dec_ms = (acc["decode"] / n) * 1000.0
    other_ms = max(0.0, per_frame_ms - pre_ms - nnie_ms - head_ms - dec_ms)

    print("============ FEAR Hi3519 分段性能基准 ============")
    print("后端           : {}{}".format(backend, "  (NNIE 一栏为 onnxruntime CPU 耗时, 非板端性能)" if backend == "onnx" else ""))
    print("帧来源         : {}".format(video_path if video_path else "随机合成 {}x{}".format(synth_height, synth_width)))
    print("帧分辨率       : {}x{}".format(frames[1].shape[1], frames[1].shape[0]))
    print("计时帧数       : {} (预热 {})".format(n, warmup))
    print("--------------------------------------------------")
    print("模板骨干(一次) : {:.2f} ms".format(template_ms))
    print("每帧端到端     : {:.2f} ms".format(per_frame_ms))
    print("  ├─ 预处理        : {:.2f} ms ({:.0f}%)".format(pre_ms, 100 * pre_ms / per_frame_ms))
    print("  ├─ NNIE 骨干(搜索): {:.2f} ms ({:.0f}%)".format(nnie_ms, 100 * nnie_ms / per_frame_ms))
    print("  ├─ CPU 头部      : {:.2f} ms ({:.0f}%)".format(head_ms, 100 * head_ms / per_frame_ms))
    print("  ├─ 解码          : {:.2f} ms ({:.0f}%)".format(dec_ms, 100 * dec_ms / per_frame_ms))
    print("  └─ 其它          : {:.2f} ms ({:.0f}%)".format(other_ms, 100 * other_ms / per_frame_ms))
    print("--------------------------------------------------")
    print("端到端 FPS     : {:.1f}".format(n / wall))
    print("==================================================")


if __name__ == "__main__":
    Fire(main)
