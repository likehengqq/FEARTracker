"""RK3588 / RV1126B 真机性能基准脚本。

对 FEAR 跟踪器做分段计时，输出端到端 FPS 以及各阶段耗时占比：

- 模板分支（初始化时跑一次）
- 每帧：CPU 预处理 / NPU 跟踪推理 / CPU 后处理 / 其它（裁剪等）

后端可选：
- ``--backend=rknn``（默认，板端用）：通过 ``rknn-toolkit-lite2`` 在 NPU 上跑，得到真实 FPS。
- ``--backend=onnx``（PC/CI 自检用）：用 ``onnxruntime`` 跑，复用完全相同的 CPU 前后处理逻辑，
  仅用于验证脚本与流水线正确性（此时“NPU”一栏是 CPU 上的 onnxruntime 耗时，不代表板端性能）。

板端用法（RK3588 多核，可用 core_mask=all；RV1126B 单核用 auto）::

    PYTHONPATH=. python evaluate/rk3588_benchmark.py \
        --template_model_path=outputs/rv1126b/fear_template_encoder.rknn \
        --track_model_path=outputs/rv1126b/fear_track.rknn \
        --video_path=assets/test.mp4 --num_frames=300 --core_mask=auto

PC 自检（用导出的 ONNX）::

    PYTHONPATH=. python evaluate/rk3588_benchmark.py --backend=onnx \
        --template_model_path=outputs/rv1126b/fear_template_encoder.onnx \
        --track_model_path=outputs/rv1126b/fear_track.onnx
"""
import sys
import time
import types
from typing import List, Optional

import numpy as np
from fire import Fire


def _install_onnx_rknnlite_shim() -> None:
    """注入一个用 onnxruntime 实现的假 rknnlite 模块，使 RK3588FEARTracker 在 PC 上也能跑。

    仅用于 --backend=onnx 的自检；板端请勿使用（用真实 rknn-toolkit-lite2）。
    """
    import onnxruntime as ort

    class _OnnxRKNNLite:
        NPU_CORE_AUTO = 0
        NPU_CORE_0 = 1
        NPU_CORE_1 = 2
        NPU_CORE_2 = 4
        NPU_CORE_0_1_2 = 7

        def __init__(self) -> None:
            self._sess: Optional[ort.InferenceSession] = None
            self._input_names: List[str] = []

        def load_rknn(self, path: str) -> int:
            self._sess = ort.InferenceSession(path)
            self._input_names = [i.name for i in self._sess.get_inputs()]
            return 0

        def init_runtime(self, core_mask: int = 0) -> int:
            return 0

        def inference(self, inputs):
            feed = {name: np.asarray(arr) for name, arr in zip(self._input_names, inputs)}
            return self._sess.run(None, feed)

        def release(self) -> None:
            self._sess = None

    rknnlite_mod = types.ModuleType("rknnlite")
    api_mod = types.ModuleType("rknnlite.api")
    api_mod.RKNNLite = _OnnxRKNNLite
    rknnlite_mod.api = api_mod
    sys.modules["rknnlite"] = rknnlite_mod
    sys.modules["rknnlite.api"] = api_mod


class _StageTimer:
    """通过包裹实例方法累计各阶段耗时（不修改 rk3588_runtime.py）。"""

    def __init__(self, tracker) -> None:
        self.tracker = tracker
        self.reset()
        self._orig_pre = tracker._preprocess_image
        self._orig_post = tracker._postprocess
        self._orig_track_infer = tracker.track_rknn.inference

        def timed_pre(image):
            t = time.perf_counter()
            out = self._orig_pre(image)
            self.acc["preprocess"] += time.perf_counter() - t
            return out

        def timed_post(*args, **kwargs):
            t = time.perf_counter()
            out = self._orig_post(*args, **kwargs)
            self.acc["postprocess"] += time.perf_counter() - t
            return out

        def timed_track_infer(inputs):
            t = time.perf_counter()
            out = self._orig_track_infer(inputs=inputs)
            self.acc["npu"] += time.perf_counter() - t
            return out

        tracker._preprocess_image = timed_pre
        tracker._postprocess = timed_post
        tracker.track_rknn.inference = timed_track_infer

    def reset(self) -> None:
        self.acc = {"preprocess": 0.0, "npu": 0.0, "postprocess": 0.0, "total": 0.0, "frames": 0}


def _load_frames(video_path: Optional[str], num_frames: int, synth_size) -> List[np.ndarray]:
    if video_path:
        import imageio.v3 as iio

        video = iio.imread(video_path)
        frames = [np.asarray(f) for f in video]
        if not frames:
            raise RuntimeError("视频没有帧: {}".format(video_path))
        # 不足则循环填充到 num_frames+1（含首帧用于初始化）
        out = []
        i = 0
        while len(out) < num_frames + 1:
            out.append(frames[i % len(frames)])
            i += 1
        return out
    h, w = synth_size
    rng = np.random.default_rng(0)
    return [rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8) for _ in range(num_frames + 1)]


def main(
    template_model_path: str = "outputs/rv1126b/fear_template_encoder.rknn",
    track_model_path: str = "outputs/rv1126b/fear_track.rknn",
    video_path: Optional[str] = "assets/test.mp4",
    initial_bbox: List[int] = [163, 53, 45, 174],
    num_frames: int = 200,
    warmup: int = 10,
    core_mask: str = "auto",
    backend: str = "rknn",
    synth_height: int = 720,
    synth_width: int = 1280,
) -> None:
    if backend == "onnx":
        _install_onnx_rknnlite_shim()
    elif backend != "rknn":
        raise ValueError("backend 仅支持 'rknn' 或 'onnx'")

    from evaluate.rk3588_runtime import RK3588FEARTracker

    frames = _load_frames(video_path, num_frames, (synth_height, synth_width))
    init_bbox = np.array(initial_bbox, dtype=np.int32)

    tracker = RK3588FEARTracker(
        template_model_path=template_model_path,
        track_model_path=track_model_path,
        core_mask=core_mask,
    )
    try:
        # 模板分支（初始化时跑一次）
        t0 = time.perf_counter()
        tracker.initialize(frames[0], init_bbox)
        template_ms = (time.perf_counter() - t0) * 1000.0

        timer = _StageTimer(tracker)

        # 预热
        for i in range(1, min(warmup, len(frames) - 1) + 1):
            tracker.update(frames[i])
        timer.reset()

        # 正式计时
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
    finally:
        tracker.release()

    acc = timer.acc
    per_frame_ms = (acc["total"] / n) * 1000.0
    pre_ms = (acc["preprocess"] / n) * 1000.0
    npu_ms = (acc["npu"] / n) * 1000.0
    post_ms = (acc["postprocess"] / n) * 1000.0
    other_ms = max(0.0, per_frame_ms - pre_ms - npu_ms - post_ms)

    print("================ FEAR 跟踪性能基准 ================")
    print("后端           : {}{}".format(backend, "  (NPU 一栏为 onnxruntime CPU 耗时, 非板端性能)" if backend == "onnx" else ""))
    print("帧来源         : {}".format(video_path if video_path else "随机合成 {}x{}".format(synth_height, synth_width)))
    print("帧分辨率       : {}x{}".format(frames[1].shape[1], frames[1].shape[0]))
    print("core_mask      : {}".format(core_mask))
    print("计时帧数       : {} (预热 {})".format(n, warmup))
    print("--------------------------------------------------")
    print("模板分支(一次) : {:.2f} ms".format(template_ms))
    print("每帧端到端     : {:.2f} ms".format(per_frame_ms))
    print("  ├─ 预处理    : {:.2f} ms ({:.0f}%)".format(pre_ms, 100 * pre_ms / per_frame_ms))
    print("  ├─ NPU 跟踪  : {:.2f} ms ({:.0f}%)".format(npu_ms, 100 * npu_ms / per_frame_ms))
    print("  ├─ 后处理    : {:.2f} ms ({:.0f}%)".format(post_ms, 100 * post_ms / per_frame_ms))
    print("  └─ 其它      : {:.2f} ms ({:.0f}%)".format(other_ms, 100 * other_ms / per_frame_ms))
    print("--------------------------------------------------")
    print("端到端 FPS     : {:.1f}".format(n / wall))
    print("仅 NPU 跟踪 FPS: {:.1f}".format(n / acc["npu"] if acc["npu"] > 0 else float("nan")))
    print("==================================================")


if __name__ == "__main__":
    Fire(main)
