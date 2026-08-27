import unittest
from unittest.mock import Mock

try:
    from model_training.train.callbacks import BestWorstMinerCallback
except ImportError as exc:  # pragma: no cover - optional in this environment
    BestWorstMinerCallback = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@unittest.skipIf(BestWorstMinerCallback is None, "pytorch-lightning is not installed")
class BestWorstMinerTests(unittest.TestCase):
    def test_from_config_disabled_when_missing_or_empty(self):
        self.assertIsNone(BestWorstMinerCallback.from_config({}))
        self.assertIsNone(BestWorstMinerCallback.from_config({"best_worst_miner": None}))
        self.assertIsNone(BestWorstMinerCallback.from_config({"best_worst_miner": {}}))

    def test_from_config_reads_sample_interval(self):
        callback = BestWorstMinerCallback.from_config(
            {
                "best_worst_miner": {
                    "metric_mode": "min",
                    "metric_to_monitor": "loss",
                    "sample_interval": 20,
                }
            }
        )
        self.assertEqual(callback.sample_interval, 20)

    def test_reuses_cached_outputs_instead_of_extra_forward(self):
        callback = BestWorstMinerCallback(sample_interval=1)
        pl_module = Mock()
        pl_module.forward = Mock(side_effect=AssertionError("should not re-forward"))
        cached = {"cls": 1}
        batch = {"image": 0}

        callback._check_score(score=1.0, pl_module=pl_module, batch=batch, cached_output=cached)
        callback._check_score(score=2.0, pl_module=pl_module, batch=batch, cached_output=cached)

        self.assertIs(callback.best_output, cached)
        self.assertIs(callback.worst_output, cached)
        pl_module.forward.assert_not_called()

    def test_skips_non_dict_validation_batches(self):
        callback = BestWorstMinerCallback()
        callback._check_score(score=0.1, pl_module=Mock(), batch=["not", "a", "dict"])
        self.assertIsNone(callback.best_score)


if __name__ == "__main__":
    unittest.main()
