"""Hi3519（NNIE）板端参考运行时：NNIE 骨干段 + CPU 头部段。

本模块给出 Hi3519 上 FEAR 跟踪的**参考实现与数据流**，与 ``evaluate/hi3519_split_export.py``
导出的三段式模型对应：

- 两个 **NNIE 骨干段**（模板 / 搜索）：纯卷积，板端跑在 NNIE（``.wk``）上；
- 一个 **CPU 头部段**：BoxTower（含互相关 MatMul、Exp、解码），板端在 ARM 上实现。

为便于在没有板子的 PC 上验证与做分段基准，默认用 ``backend="onnx"``（onnxruntime）模拟三段；
板端 C++ 实现见 ``evaluate/hi3519_cpp``。预处理 / 裁剪 / 解码等 CPU 数学与训练端、CoreML、RKNN
版本完全一致。

边界框格式统一为 ``[x, y, width, height]``（像素），输入帧为 RGB ``np.ndarray``。
"""
from typing import Any, Dict, List, Optional, Tuple

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

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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


def _get_extended_crop(image, bbox, crop_size, offset, padding_value=None):
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
        crop, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=tuple(float(v) for v in padding_value[:3]),
    )
    padded_bbox = np.array([bbox[0] - context[0], bbox[1] - context[1], bbox[2], bbox[3]], dtype=np.int32)
    padded_bbox = _ensure_bbox_boundaries(padded_bbox, padded_crop.shape[:2])
    resized_bbox = _resize_bbox(padded_bbox, padded_crop.shape[:2], crop_size)
    resized_crop = cv2.resize(padded_crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
    return resized_crop, resized_bbox, context


def _make_grid(score_size: int, total_stride: int, instance_size: int):
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
        raise ValueError("期望 4D 输出，得到 shape {}".format(output.shape))
    if output.shape[1] == channels:
        return output
    if output.shape[-1] == channels:
        return np.transpose(output, (0, 3, 1, 2))
    raise ValueError("无法把输出 shape {} 解释为 {} 通道的 NCHW".format(output.shape, channels))


class OnnxBackend:
    """用 onnxruntime 模拟 NNIE 骨干段与 CPU 头部段（PC 验证 / 基准用）。

    板端请改用 NNIE(.wk) 跑骨干段、ARM 实现头部段（见 evaluate/hi3519_cpp）。
    """

    def __init__(self, template_backbone_path, search_backbone_path, head_path):
        import onnxruntime as ort

        self._tmpl = ort.InferenceSession(template_backbone_path)
        self._search = ort.InferenceSession(search_backbone_path)
        self._head = ort.InferenceSession(head_path)

    def backbone_template(self, image_nchw: np.ndarray) -> np.ndarray:
        name = self._tmpl.get_inputs()[0].name
        return self._tmpl.run(None, {name: image_nchw})[0]

    def backbone_search(self, image_nchw: np.ndarray) -> np.ndarray:
        name = self._search.get_inputs()[0].name
        return self._search.run(None, {name: image_nchw})[0]

    def head(self, search_features: np.ndarray, template_features: np.ndarray) -> List[np.ndarray]:
        names = [i.name for i in self._head.get_inputs()]
        return self._head.run(None, {names[0]: search_features, names[1]: template_features})


class Hi3519FEARTracker:
    """Hi3519 上的 FEAR 跟踪器：NNIE 骨干段 + CPU 头部段。"""

    def __init__(
        self,
        template_backbone_path: str,
        search_backbone_path: str,
        head_path: str,
        backend: str = "onnx",
        backend_impl: Optional[Any] = None,
        tracker_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.tracker_config = DEFAULT_TRACKER_CONFIG.copy()
        if tracker_config is not None:
            self.tracker_config.update(tracker_config)

        if backend_impl is not None:
            self.backend = backend_impl
        elif backend == "onnx":
            self.backend = OnnxBackend(template_backbone_path, search_backbone_path, head_path)
        else:
            raise ValueError("backend 仅支持 'onnx'；板端请通过 backend_impl 注入 NNIE 实现")

        self.grid_x, self.grid_y = _make_grid(
            self.tracker_config["score_size"],
            self.tracker_config["total_stride"],
            self.tracker_config["instance_size"],
        )
        self.window = self._get_tracking_window(self.tracker_config["windowing"], self.tracker_config["score_size"])
        self.bbox = None
        self.prev_size = None
        self.mean_color = None
        self.template_features = None

    @staticmethod
    def _get_tracking_window(windowing: str, score_size: int) -> np.ndarray:
        if windowing == "cosine":
            return np.outer(np.hanning(score_size), np.hanning(score_size)).astype(np.float32)
        return np.ones((score_size, score_size), dtype=np.float32)

    @staticmethod
    def _preprocess_image(image: np.ndarray) -> np.ndarray:
        image = image[:, :, :3].astype(np.float32) / 255.0
        image = (image - _MEAN) / _STD
        image = np.transpose(image, (2, 0, 1))[np.newaxis, ...]
        return np.ascontiguousarray(image, dtype=np.float32)

    def initialize(self, image: np.ndarray, bbox: np.ndarray) -> None:
        bbox = _clamp_bbox(np.asarray(bbox, dtype=np.int32), image.shape[:2])
        self.bbox = bbox
        self.mean_color = np.mean(image, axis=(0, 1))
        template_crop, _, _ = _get_extended_crop(
            image=image, bbox=bbox,
            offset=self.tracker_config["template_bbox_offset"],
            crop_size=self.tracker_config["template_size"],
        )
        template_input = self._preprocess_image(template_crop)
        self.template_features = np.ascontiguousarray(self.backend.backbone_template(template_input), dtype=np.float32)

    def update(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        if self.template_features is None or self.bbox is None:
            raise RuntimeError("update() 前必须先 initialize()")
        search_crop, search_bbox, context = _get_extended_crop(
            image=image, bbox=self.bbox,
            crop_size=self.tracker_config["instance_size"],
            offset=self.tracker_config["search_context"],
            padding_value=self.mean_color,
        )
        self.prev_size = search_bbox[2:]
        pred_bbox, score = self.track(search_crop)
        pred_bbox = self._rescale_bbox(pred_bbox, context)
        pred_bbox = _clamp_bbox(pred_bbox, image.shape[:2])
        self.bbox = pred_bbox
        return {"bbox": pred_bbox, "score": score}

    def track(self, search_crop: np.ndarray) -> Tuple[np.ndarray, float]:
        search_input = self._preprocess_image(search_crop)
        search_features = np.ascontiguousarray(self.backend.backbone_search(search_input), dtype=np.float32)
        outputs = self.backend.head(search_features, self.template_features)
        if len(outputs) != 2:
            raise RuntimeError("头部段返回 {} 个输出，期望 2".format(len(outputs)))
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

    def _confidence_postprocess(self, cls_score, regression_map):
        if not self.tracker_config.get("smooth", False):
            return cls_score, None
        pred_location = np.stack(
            [self.grid_x - regression_map[0], self.grid_y - regression_map[1],
             self.grid_x + regression_map[2], self.grid_y + regression_map[3]], axis=0,
        )
        prev_w, prev_h = self.prev_size
        pred_w = pred_location[2] - pred_location[0]
        pred_h = pred_location[3] - pred_location[1]
        s_c = _limit(_squared_size(pred_w, pred_h) / _squared_size(prev_w, prev_h))
        r_c = _limit((prev_w / prev_h) / (pred_w / pred_h))
        penalty = np.exp(-(r_c * s_c - 1) * self.tracker_config["penalty_k"])
        pscore = penalty * cls_score
        pscore = pscore * (1 - self.tracker_config["window_influence"]) + self.window * self.tracker_config["window_influence"]
        return pscore, penalty

    def _decode(self, regression_map, classification_map):
        pred_location = np.stack(
            [self.grid_x - regression_map[0], self.grid_y - regression_map[1],
             self.grid_x + regression_map[2], self.grid_y + regression_map[3]], axis=0,
        )
        flat_index = int(np.argmax(classification_map))
        r_max, c_max = np.unravel_index(flat_index, classification_map.shape)
        x1, y1, x2, y2 = pred_location[:, r_max, c_max]
        return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32), (r_max, c_max)

    def _postprocess_bbox(self, bbox, cls_score, pred_coords, penalty):
        if not self.tracker_config.get("smooth", False):
            return bbox
        r_max, c_max = pred_coords
        lr = penalty[r_max, c_max] * cls_score[r_max, c_max] * self.tracker_config["lr"]
        pred_size = bbox[2:] * lr
        prev_size = self.prev_size * (1 - lr)
        smoothed_size = prev_size + lr * (pred_size + prev_size)
        return np.array([bbox[0], bbox[1], smoothed_size[0], smoothed_size[1]], dtype=np.float32)

    def _rescale_bbox(self, bbox, context):
        w_scale = context[2] / self.tracker_config["instance_size"]
        h_scale = context[3] / self.tracker_config["instance_size"]
        return np.array(
            [round(bbox[0] * w_scale + context[0]), round(bbox[1] * h_scale + context[1]),
             max(3, round(bbox[2] * w_scale)), max(3, round(bbox[3] * h_scale))], dtype=np.int32,
        )
