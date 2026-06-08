from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


DEFAULT_TRACKER_CONFIG = {
    "penalty_k": 0.062,
    "window_influence": 0.38,
    "lr": 0.765,
    "windowing": "cosine",
    "total_stride": 16,
    "score_size": 16,
    "template_bbox_offset": 0.2,
    "search_context": 2,
    "instance_size": 256,
    "template_size": 128,
    "smooth": False,
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _limit(radius: np.ndarray) -> np.ndarray:
    return np.maximum(radius, 1.0 / radius)


def _squared_size(w: np.ndarray, h: np.ndarray) -> np.ndarray:
    pad = (w + h) * 0.5
    return np.sqrt((w + pad) * (h + pad))


def _extend_bbox(bbox: np.ndarray, offset: float) -> np.ndarray:
    x, y, w, h = bbox
    return np.array(
        [x - w * offset, y - h * offset, w * (1.0 + 2.0 * offset), h * (1.0 + 2.0 * offset)],
        dtype=np.int32,
    )


def _ensure_bbox_boundaries(bbox: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
    x1, y1, w, h = bbox
    x1 = min(max(0, x1), image_shape[1])
    y1 = min(max(0, y1), image_shape[0])
    x2 = min(max(0, x1 + w), image_shape[1])
    y2 = min(max(0, y1 + h), image_shape[0])
    return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.int32)


def _clamp_bbox(bbox: np.ndarray, image_shape: Tuple[int, int], min_side: int = 3) -> np.ndarray:
    bbox = _ensure_bbox_boundaries(np.asarray(bbox, dtype=np.int32), image_shape)
    x, y, w, h = bbox
    img_h, img_w = image_shape[:2]
    if w < min_side:
        w = min_side
        x -= max(0, x + w - img_w)
    if h < min_side:
        h = min_side
        y -= max(0, y + h - img_h)
    return np.array([x, y, w, h], dtype=np.int32)


def _resize_bbox(bbox: np.ndarray, from_shape: Tuple[int, int], out_size: int) -> np.ndarray:
    height, width = from_shape[:2]
    scale_x = float(out_size) / float(width)
    scale_y = float(out_size) / float(height)
    x, y, w, h = bbox.astype(np.float32)
    return np.array([x * scale_x, y * scale_y, w * scale_x, h * scale_y], dtype=np.float32)


def _get_extended_crop(
    image: np.ndarray,
    bbox: np.ndarray,
    crop_size: int,
    offset: float,
    padding_value: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if padding_value is None:
        padding_value = np.mean(image, axis=(0, 1))

    context = _extend_bbox(np.asarray(bbox, dtype=np.int32), offset)
    pad_left = max(-int(context[0]), 0)
    pad_top = max(-int(context[1]), 0)
    pad_right = max(int(context[0] + context[2] - image.shape[1]), 0)
    pad_bottom = max(int(context[1] + context[3] - image.shape[0]), 0)

    crop = image[
        context[1] + pad_top: context[1] + context[3] - pad_bottom,
        context[0] + pad_left: context[0] + context[2] - pad_right,
    ]
    padded_crop = cv2.copyMakeBorder(
        crop,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=tuple(float(v) for v in padding_value[:3]),
    )

    padded_bbox = np.array([bbox[0] - context[0], bbox[1] - context[1], bbox[2], bbox[3]], dtype=np.int32)
    padded_bbox = _ensure_bbox_boundaries(padded_bbox, padded_crop.shape[:2])
    resized_bbox = _resize_bbox(padded_bbox, padded_crop.shape[:2], crop_size)
    resized_crop = cv2.resize(padded_crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
    return resized_crop, resized_bbox, context


def _make_grid(score_size: int, total_stride: int, instance_size: int) -> Tuple[np.ndarray, np.ndarray]:
    x, y = np.meshgrid(
        np.arange(0, score_size) - np.floor(float(score_size // 2)),
        np.arange(0, score_size) - np.floor(float(score_size // 2)),
    )
    grid_x = x * total_stride + instance_size // 2
    grid_y = y * total_stride + instance_size // 2
    return grid_x.astype(np.float32), grid_y.astype(np.float32)


def _nchw(output: np.ndarray, channels: int) -> np.ndarray:
    output = np.asarray(output)
    if output.ndim == 3 and channels == 1:
        return output[:, np.newaxis, :, :]
    if output.ndim != 4:
        raise ValueError("Expected a 4D RKNN output, got shape {}".format(output.shape))
    if output.shape[1] == channels:
        return output
    if output.shape[-1] == channels:
        return np.transpose(output, (0, 3, 1, 2))
    raise ValueError("Cannot interpret output shape {} as {}-channel NCHW".format(output.shape, channels))


class RK3588FEARTracker:
    """
    FEAR tracker backed by two RKNNLite models on RK3588.

    Input frames must be RGB numpy arrays. Bounding boxes use [x, y, width, height].
    """

    def __init__(
        self,
        template_model_path: str,
        track_model_path: str,
        core_mask: str = "auto",
        tracker_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            from rknnlite.api import RKNNLite
        except ImportError as exc:
            raise ImportError("rknn-toolkit-lite2 is required on the RK3588 runtime device.") from exc

        self.RKNNLite = RKNNLite
        self.tracker_config = DEFAULT_TRACKER_CONFIG.copy()
        if tracker_config is not None:
            self.tracker_config.update(tracker_config)

        self.template_rknn = self._load_rknn(template_model_path, core_mask)
        self.track_rknn = self._load_rknn(track_model_path, core_mask)
        self.grid_x, self.grid_y = _make_grid(
            self.tracker_config["score_size"],
            self.tracker_config["total_stride"],
            self.tracker_config["instance_size"],
        )
        self.window = self._get_tracking_window(
            self.tracker_config["windowing"], self.tracker_config["score_size"]
        )
        self.bbox = None
        self.prev_size = None
        self.mean_color = None
        self.template_features = None

    def _core_mask_value(self, core_mask: str) -> int:
        mapping = {
            "auto": self.RKNNLite.NPU_CORE_AUTO,
            "0": self.RKNNLite.NPU_CORE_0,
            "1": self.RKNNLite.NPU_CORE_1,
            "2": self.RKNNLite.NPU_CORE_2,
            "all": self.RKNNLite.NPU_CORE_0_1_2,
        }
        if core_mask not in mapping:
            raise ValueError("Unsupported core_mask '{}'. Use one of {}.".format(core_mask, sorted(mapping)))
        return mapping[core_mask]

    def _load_rknn(self, model_path: str, core_mask: str):
        rknn = self.RKNNLite()
        ret = rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError("Failed to load RKNN model {} with code {}".format(model_path, ret))
        ret = rknn.init_runtime(core_mask=self._core_mask_value(core_mask))
        if ret != 0:
            raise RuntimeError("Failed to initialize RKNN runtime for {} with code {}".format(model_path, ret))
        return rknn

    @staticmethod
    def _get_tracking_window(windowing: str, score_size: int) -> np.ndarray:
        if windowing == "cosine":
            return np.outer(np.hanning(score_size), np.hanning(score_size)).astype(np.float32)
        return np.ones((score_size, score_size), dtype=np.float32)

    @staticmethod
    def _preprocess_image(image: np.ndarray) -> np.ndarray:
        image = image[:, :, :3].astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        image = np.transpose(image, (2, 0, 1))[np.newaxis, ...]
        return np.ascontiguousarray(image, dtype=np.float32)

    def initialize(self, image: np.ndarray, bbox: np.ndarray) -> None:
        bbox = _clamp_bbox(np.asarray(bbox, dtype=np.int32), image.shape[:2])
        self.bbox = bbox
        self.mean_color = np.mean(image, axis=(0, 1))

        template_crop, _, _ = _get_extended_crop(
            image=image,
            bbox=bbox,
            offset=self.tracker_config["template_bbox_offset"],
            crop_size=self.tracker_config["template_size"],
        )
        template_input = self._preprocess_image(template_crop)
        outputs = self.template_rknn.inference(inputs=[template_input])
        if not outputs:
            raise RuntimeError("Template encoder returned no outputs.")
        self.template_features = np.ascontiguousarray(outputs[0], dtype=np.float32)

    def update(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        if self.template_features is None or self.bbox is None:
            raise RuntimeError("Tracker must be initialized before update().")

        search_crop, search_bbox, padded_bbox = _get_extended_crop(
            image=image,
            bbox=self.bbox,
            crop_size=self.tracker_config["instance_size"],
            offset=self.tracker_config["search_context"],
            padding_value=self.mean_color,
        )
        self.prev_size = search_bbox[2:]
        pred_bbox, score = self.track(search_crop)
        pred_bbox = self._rescale_bbox(pred_bbox, padded_bbox)
        pred_bbox = _clamp_bbox(pred_bbox, image.shape[:2])
        self.bbox = pred_bbox
        return {"bbox": pred_bbox, "score": score}

    def track(self, search_crop: np.ndarray) -> Tuple[np.ndarray, float]:
        search_input = self._preprocess_image(search_crop)
        outputs = self.track_rknn.inference(inputs=[search_input, self.template_features])
        if len(outputs) != 2:
            raise RuntimeError("Tracking model returned {} outputs, expected 2.".format(len(outputs)))

        regression_map = _nchw(outputs[0], channels=4).astype(np.float32)
        cls_logits = _nchw(outputs[1], channels=1).astype(np.float32)
        return self._postprocess(regression_map=regression_map, cls_logits=cls_logits)

    def _postprocess(self, regression_map: np.ndarray, cls_logits: np.ndarray) -> Tuple[np.ndarray, float]:
        cls_score = _sigmoid(cls_logits)[0, 0]
        classification_map, penalty = self._confidence_postprocess(cls_score, regression_map[0])
        bbox, pred_coords = self._decode(regression_map[0], classification_map)
        r_max, c_max = pred_coords
        bbox = self._postprocess_bbox(bbox, cls_score, pred_coords, penalty)
        return bbox, float(cls_score[r_max, c_max])

    def _confidence_postprocess(
        self, cls_score: np.ndarray, regression_map: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not self.tracker_config.get("smooth", False):
            return cls_score, None

        pred_location = np.stack(
            [
                self.grid_x - regression_map[0],
                self.grid_y - regression_map[1],
                self.grid_x + regression_map[2],
                self.grid_y + regression_map[3],
            ],
            axis=0,
        )
        prev_w, prev_h = self.prev_size
        pred_w = pred_location[2] - pred_location[0]
        pred_h = pred_location[3] - pred_location[1]
        s_c = _limit(_squared_size(pred_w, pred_h) / _squared_size(prev_w, prev_h))
        r_c = _limit((prev_w / prev_h) / (pred_w / pred_h))
        penalty = np.exp(-(r_c * s_c - 1) * self.tracker_config["penalty_k"])
        pscore = penalty * cls_score
        pscore = pscore * (1 - self.tracker_config["window_influence"]) + (
            self.window * self.tracker_config["window_influence"]
        )
        return pscore, penalty

    def _decode(self, regression_map: np.ndarray, classification_map: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        pred_location = np.stack(
            [
                self.grid_x - regression_map[0],
                self.grid_y - regression_map[1],
                self.grid_x + regression_map[2],
                self.grid_y + regression_map[3],
            ],
            axis=0,
        )
        flat_index = int(np.argmax(classification_map))
        r_max, c_max = np.unravel_index(flat_index, classification_map.shape)
        x1, y1, x2, y2 = pred_location[:, r_max, c_max]
        return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32), (r_max, c_max)

    def _postprocess_bbox(
        self,
        bbox: np.ndarray,
        cls_score: np.ndarray,
        pred_coords: Tuple[int, int],
        penalty: Optional[np.ndarray],
    ) -> np.ndarray:
        if not self.tracker_config.get("smooth", False):
            return bbox

        r_max, c_max = pred_coords
        lr = penalty[r_max, c_max] * cls_score[r_max, c_max] * self.tracker_config["lr"]
        pred_size = bbox[2:] * lr
        prev_size = self.prev_size * (1 - lr)
        smoothed_size = prev_size + lr * (pred_size + prev_size)
        return np.array([bbox[0], bbox[1], smoothed_size[0], smoothed_size[1]], dtype=np.float32)

    def _rescale_bbox(self, bbox: np.ndarray, padded_box: np.ndarray) -> np.ndarray:
        w_scale = padded_box[2] / self.tracker_config["instance_size"]
        h_scale = padded_box[3] / self.tracker_config["instance_size"]
        return np.array(
            [
                round(bbox[0] * w_scale + padded_box[0]),
                round(bbox[1] * h_scale + padded_box[1]),
                max(3, round(bbox[2] * w_scale)),
                max(3, round(bbox[3] * h_scale)),
            ],
            dtype=np.int32,
        )

    def release(self) -> None:
        self.template_rknn.release()
        self.track_rknn.release()

    def __enter__(self) -> "RK3588FEARTracker":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()
