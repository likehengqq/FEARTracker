import unittest

import numpy as np

from model_training.dataset.bbox_parse import parse_bbox


class ParseBboxTests(unittest.TestCase):
    def test_tuple_and_list(self):
        np.testing.assert_array_equal(parse_bbox((1, 2, 3, 4)), np.array([1, 2, 3, 4], dtype=np.float32))
        np.testing.assert_array_equal(parse_bbox([10, 20, 30, 40]), np.array([10, 20, 30, 40], dtype=np.float32))

    def test_numpy_array(self):
        src = np.array([5, 6, 7, 8], dtype=np.int32)
        parsed = parse_bbox(src)
        np.testing.assert_array_equal(parsed, np.array([5, 6, 7, 8], dtype=np.float32))

    def test_csv_strings(self):
        expected = np.array([163, 53, 45, 174], dtype=np.float32)
        for raw in (
            "[163, 53, 45, 174]",
            "(163, 53, 45, 174)",
            "163, 53, 45, 174",
            "163,53,45,174",
        ):
            np.testing.assert_array_equal(parse_bbox(raw), expected)

    def test_invalid_string(self):
        with self.assertRaises(ValueError):
            parse_bbox("1, 2, 3")
        with self.assertRaises(ValueError):
            parse_bbox("")


if __name__ == "__main__":
    unittest.main()
