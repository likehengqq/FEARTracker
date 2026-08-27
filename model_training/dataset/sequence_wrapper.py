from typing import Any, Dict

from torch.utils.data import Dataset


def dummy_collate(batch: Any) -> Any:
    return batch


class SequenceDatasetWrapper(Dataset):
    def __init__(self, dataset_name: str, dataset: Dataset):
        self.dataset_name = dataset_name
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __str__(self):
        return self.dataset_name

    def __getitem__(self, index: int):
        image_files, annotations = self.dataset[index]
        return image_files, annotations, self.dataset_name

    def get_collate_fn(self) -> Any:
        return dummy_collate

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        from got10k.datasets import GOT10k, NfS, VOT

        datasets = {
            "nfs": NfS,
            "got10k": GOT10k,
            "vot": VOT,
        }
        dataset_name = config.pop("name")
        dataset = datasets[dataset_name](**config)
        return cls(dataset_name=dataset_name, dataset=dataset)
