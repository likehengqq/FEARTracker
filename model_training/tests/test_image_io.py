import os
import tempfile
import unittest

import numpy as np

from model_training.dataset.image_io import read_img


class ReadImgTests(unittest.TestCase):
    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            read_img("/tmp/this-fear-image-does-not-exist.png")

    def test_returns_rgb(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("opencv is not installed")

        bgr = np.zeros((8, 8, 3), dtype=np.uint8)
        bgr[:] = (0, 0, 255)
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
        self.assertTrue(cv2.imwrite(path, bgr))
        rgb = read_img(path)
        self.assertEqual(rgb.shape, (8, 8, 3))
        self.assertEqual(tuple(rgb[0, 0]), (255, 0, 0))


if __name__ == "__main__":
    unittest.main()
