import cv2
import numpy as np


def read_img(path: str) -> np.ndarray:
    """
    Args:
        path: image path
    Returns: RGB image
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError("failed to read image: %s" % path)
    # cvtColor already allocates a new array; an extra copy was wasted work.
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
