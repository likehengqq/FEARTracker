"""Hi3519 参考运行时的视频跟踪 demo（NNIE 骨干 + CPU 头部）。

默认用 onnx 后端（onnxruntime 模拟三段），便于在 PC 上验证整条流水线与 PyTorch 一致。
板端请用 evaluate/hi3519_cpp 的 C++ 实现（NNIE 跑 .wk 骨干 + ARM 头部）。

用法::

    PYTHONPATH=. python evaluate/hi3519_demo_video.py \
        --output_dir=outputs/hi3519_split \
        --video_path=assets/test.mp4 \
        --output_path=outputs/hi3519_split/test.mp4
"""
import os
from typing import List

import cv2
import imageio.v3 as iio
import numpy as np
from fire import Fire

from evaluate.hi3519_runtime import Hi3519FEARTracker


def draw_bbox(image: np.ndarray, bbox: np.ndarray, width: int = 5) -> np.ndarray:
    image = image.copy()
    x, y, w, h = np.asarray(bbox).astype(int)
    return cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), width)


def main(
    output_dir: str = "outputs/hi3519_split",
    initial_bbox: List[int] = [163, 53, 45, 174],
    video_path: str = "assets/test.mp4",
    output_path: str = "outputs/hi3519_split/test.mp4",
    backend: str = "onnx",
) -> None:
    tracker = Hi3519FEARTracker(
        template_backbone_path=os.path.join(output_dir, "hi3519_backbone_template.onnx"),
        search_backbone_path=os.path.join(output_dir, "hi3519_backbone_search.onnx"),
        head_path=os.path.join(output_dir, "hi3519_head.onnx"),
        backend=backend,
    )

    frames, metadata = iio.imread(video_path), iio.immeta(video_path, exclude_applied=False)
    init_bbox = np.array(initial_bbox, dtype=np.int32)

    tracked_bboxes = [init_bbox]
    tracker.initialize(frames[0], init_bbox)
    for frame in frames[1:]:
        tracked_bboxes.append(tracker.update(frame)["bbox"])

    visualized = [draw_bbox(f, b) for f, b in zip(frames, tracked_bboxes)]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    iio.imwrite(output_path, visualized, fps=metadata["fps"])
    print("frames tracked:", len(tracked_bboxes))
    print("first bbox:", tracked_bboxes[0], "last bbox:", tracked_bboxes[-1])
    print("output written:", output_path)


if __name__ == "__main__":
    Fire(main)
