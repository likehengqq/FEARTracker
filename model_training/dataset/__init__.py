from typing import Any, Dict


def dummy_collate(batch: Any) -> Any:
    return batch


def get_tracking_dataset(config: Dict):
    from .siam_dataset import SiameseTrackingDataset

    datasets = {
        "siam": SiameseTrackingDataset,
    }
    cls = datasets[config["dataset"]["dataset_type"]]
    return cls.from_config(config)


def get_tracking_datasets(config):
    from torch.utils.data import ConcatDataset

    from model_training.utils.logger import create_logger

    from .sequence_wrapper import SequenceDatasetWrapper

    logger = create_logger(__name__)
    train_datasets = []
    for dataset_config in config["train"]["datasets"]:
        ds = get_tracking_dataset(dict(dataset=dataset_config, tracker=config["tracker"]))
        logger.info("Train dataset %s %d", str(ds), len(ds))
        train_datasets.append(ds)

    val_datasets = []
    for dataset_config in config["val"]["datasets"]:
        ds = SequenceDatasetWrapper.from_config(dataset_config)
        logger.info("Valid dataset %s %d", str(ds), len(ds))
        val_datasets.append(ds)
    return ConcatDataset(train_datasets), ConcatDataset(val_datasets)


def __getattr__(name: str):
    if name == "SiameseTrackingDataset":
        from .siam_dataset import SiameseTrackingDataset

        return SiameseTrackingDataset
    if name == "TrackingDataset":
        from .tracking_dataset import TrackingDataset

        return TrackingDataset
    if name == "SequenceDatasetWrapper":
        from .sequence_wrapper import SequenceDatasetWrapper

        return SequenceDatasetWrapper
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


__all__ = [
    "SiameseTrackingDataset",
    "TrackingDataset",
    "SequenceDatasetWrapper",
    "get_tracking_dataset",
    "get_tracking_datasets",
    "dummy_collate",
]
