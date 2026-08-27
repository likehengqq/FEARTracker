import unittest

from model_training.train.dataloader_utils import build_dataloader_kwargs, resolve_num_workers, worker_init_fn


def _identity_collate(batch):
    return batch


class DataloaderUtilsTests(unittest.TestCase):
    def test_val_workers_are_not_capped_by_batch_size(self):
        config = {"num_workers": 8, "batch_size": {"train": 128, "val": 1}}
        self.assertEqual(resolve_num_workers(config, "val"), 8)
        self.assertEqual(resolve_num_workers(config, "train"), 8)

    def test_split_num_workers_override(self):
        config = {"num_workers": 8, "val": {"num_workers": 2}}
        self.assertEqual(resolve_num_workers(config, "val"), 2)
        self.assertEqual(resolve_num_workers(config, "train"), 8)

    def test_prefetch_enabled_when_multiprocessed(self):
        kwargs = build_dataloader_kwargs(
            batch_size=128,
            num_workers=8,
            shuffle=True,
            sampler=None,
            drop_last=True,
            collate_fn=_identity_collate,
            prefetch_factor=4,
        )
        self.assertEqual(kwargs["num_workers"], 8)
        self.assertNotIn("persistent_workers", kwargs)
        self.assertEqual(kwargs["prefetch_factor"], 4)
        self.assertIs(kwargs["worker_init_fn"], worker_init_fn)
        self.assertTrue(kwargs["pin_memory"])

    def test_persistent_workers_opt_in(self):
        kwargs = build_dataloader_kwargs(
            batch_size=128,
            num_workers=8,
            shuffle=True,
            sampler=None,
            drop_last=True,
            collate_fn=_identity_collate,
            persistent_workers=True,
        )
        self.assertTrue(kwargs["persistent_workers"])

    def test_zero_workers_skip_persistent_settings(self):
        kwargs = build_dataloader_kwargs(
            batch_size=1,
            num_workers=0,
            shuffle=False,
            sampler=None,
            drop_last=False,
            collate_fn=_identity_collate,
        )
        self.assertEqual(kwargs["num_workers"], 0)
        self.assertNotIn("persistent_workers", kwargs)
        self.assertNotIn("prefetch_factor", kwargs)
        self.assertNotIn("worker_init_fn", kwargs)


if __name__ == "__main__":
    unittest.main()
