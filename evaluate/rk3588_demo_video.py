import os
from typing import List

import cv2
import imageio.v3 as iio
import numpy as np
from fire import Fire

from evaluate.rk3588_runtime import RK3588FEARTracker


def draw_bbox(image: np.ndarray, bbox: np.ndarray, width: int = 5) -> np.ndarray:
    image = image.copy()
    x, y, w, h = bbox.astype(int)
    return cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), width)


def track_video(
    tracker: RK3588FEARTracker,
    frames: List[np.ndarray],
    initial_bbox: np.ndarray,
) -> List[np.ndarray]:
    tracked_bboxes = [initial_bbox]
    tracker.initialize(frames[0], initial_bbox)
    for frame in frames[1:]:
        tracked_bboxes.append(tracker.update(frame)["bbox"])
    return tracked_bboxes


def main(
    template_model_path: str = "outputs/rk3588/fear_template_encoder.rknn",
    track_model_path: str = "outputs/rk3588/fear_track.rknn",
    initial_bbox: List[int] = [163, 53, 45, 174],
    video_path: str = "assets/test.mp4",
    output_path: str = "outputs/rk3588/test.mp4",
    core_mask: str = "auto",
) -> None:
    frames, metadata = iio.imread(video_path), iio.immeta(video_path, exclude_applied=False)
    initial_bbox_array = np.array(initial_bbox, dtype=np.int32)

    with RK3588FEARTracker(
        template_model_path=template_model_path,
        track_model_path=track_model_path,
        core_mask=core_mask,
    ) as tracker:
        tracked_bboxes = track_video(tracker, frames, initial_bbox_array)

    visualized_video = [draw_bbox(frame, bbox) for frame, bbox in zip(frames, tracked_bboxes)]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    iio.imwrite(output_path, visualized_video, fps=metadata["fps"])


if __name__ == "__main__":
    Fire(main)
