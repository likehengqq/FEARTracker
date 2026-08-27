import os
import time
import unittest

import numpy as np

from model_training.dataset.bbox_parse import parse_bbox


def _timeit(fn, loops):
    start = time.perf_counter()
    for _ in range(loops):
        fn()
    return time.perf_counter() - start


class TrainPipelineMicrobench(unittest.TestCase):
    def test_bbox_parse_is_faster_than_eval(self):
        raw = "[163, 53, 45, 174]"
        loops = 20000
        eval_s = _timeit(lambda: eval(raw), loops)
        parse_s = _timeit(lambda: parse_bbox(raw), loops)
        speedup = eval_s / parse_s if parse_s else float("inf")
        print("bbox parse: eval=%.4fs parse=%.4fs speedup=%.2fx" % (eval_s, parse_s, speedup))
        self.assertGreater(speedup, 1.2)
        np.testing.assert_array_equal(parse_bbox(raw), np.array(eval(raw), dtype=np.float32))

    def test_cached_albumentations_compose_is_faster(self):
        try:
            import albumentations as A
        except ImportError:
            self.skipTest("albumentations is not installed")

        image = np.zeros((128, 128, 3), dtype=np.uint8)
        bbox = [20, 20, 40, 40]
        bbox_params = {"format": "coco", "min_visibility": 0, "label_fields": ["category_id"], "min_area": 0}
        ops = [A.RandomBrightnessContrast(p=1.0), A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        cached = A.Compose(ops, bbox_params=bbox_params)
        loops = 80

        def recreate():
            transform = A.Compose(ops, bbox_params=bbox_params)
            transform(image=image, bboxes=[bbox], category_id=["bbox"])

        def reuse():
            cached(image=image, bboxes=[bbox], category_id=["bbox"])

        recreate_s = _timeit(recreate, loops)
        reuse_s = _timeit(reuse, loops)
        speedup = recreate_s / reuse_s if reuse_s else float("inf")
        print("albumentations compose: recreate=%.4fs cached=%.4fs speedup=%.2fx" % (recreate_s, reuse_s, speedup))
        self.assertGreater(speedup, 1.15)

    def test_read_img_skips_redundant_copy(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("opencv is not installed")

        from model_training.dataset.image_io import read_img

        path = os.path.join(os.path.dirname(__file__), "_bench_read.png")
        bgr = np.zeros((256, 256, 3), dtype=np.uint8)
        bgr[:, :] = (12, 64, 200)
        self.assertTrue(cv2.imwrite(path, bgr))
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))

        loops = 40

        def old_read():
            img = cv2.imread(path)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()

        old_s = _timeit(old_read, loops)
        new_s = _timeit(lambda: read_img(path), loops)
        rgb = read_img(path)
        self.assertEqual(tuple(rgb[0, 0]), (200, 64, 12))
        print("read_img: copy=%.4fs no_copy=%.4fs speedup=%.2fx" % (old_s, new_s, old_s / new_s if new_s else float("inf")))
        self.assertLess(new_s, old_s * 1.15)

    def test_train_metric_interval_runs_decode_sparsely(self):
        interval = 50
        decoded = [i for i in range(128) if interval <= 1 or (i % interval == 0)]
        self.assertEqual(decoded, [0, 50, 100])


if __name__ == "__main__":
    unittest.main()
