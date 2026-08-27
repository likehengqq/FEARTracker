from typing import Any

import numpy as np


def parse_bbox(bbox: Any) -> np.ndarray:
    """
    Parse a COCO-style xywh bbox from CSV cells, lists, or arrays.

    Avoids `eval()` which is both slower and unsafe on annotation strings.
    """
    if isinstance(bbox, np.ndarray):
        return bbox.astype(np.float32, copy=False)
    if isinstance(bbox, (list, tuple)):
        return np.asarray(bbox, dtype=np.float32)
    if isinstance(bbox, str):
        text = bbox.strip()
        if not text:
            raise ValueError("empty bbox string")
        if text[0] in "([":
            text = text[1:]
        if text and text[-1] in ")]":
            text = text[:-1]
        values = np.fromstring(text, sep=",", dtype=np.float32)
        if values.size != 4:
            values = np.fromstring(text.replace(" ", ","), sep=",", dtype=np.float32)
        if values.size != 4:
            raise ValueError("bbox must contain 4 numbers, got %r" % bbox)
        return values
    return np.asarray(bbox, dtype=np.float32)
