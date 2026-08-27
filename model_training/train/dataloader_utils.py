from typing import Any, Callable, Dict, Optional

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def worker_init_fn(_worker_id: int) -> None:
    """Limit OpenCV threads per DataLoader worker to avoid CPU oversubscription."""
    if cv2 is None:
        return
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)


def resolve_num_workers(config: Dict[str, Any], loader_name: str) -> int:
    """
    Resolve DataLoader workers for a split.

    Validation uses batch_size=1, so workers must not be capped by batch size.
    A per-split `num_workers` key (train/val) wins over the global value.
    """
    split_config = config.get(loader_name) or {}
    if isinstance(split_config, dict) and "num_workers" in split_config:
        num_workers = split_config["num_workers"]
    else:
        num_workers = config.get("num_workers", 0)
    return max(0, int(num_workers))


def build_dataloader_kwargs(
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    sampler: Any,
    drop_last: bool,
    collate_fn: Callable,
    pin_memory: bool = True,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: Optional[int] = 2,
) -> Dict[str, Any]:
    num_workers = max(0, int(num_workers))
    kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "collate_fn": collate_fn,
    }
    if num_workers > 0:
        kwargs["worker_init_fn"] = worker_init_fn
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(prefetch_factor)
        # persistent_workers is opt-in: this project resamples the train set
        # after each epoch, and live workers would keep stale samples.
        if persistent_workers:
            kwargs["persistent_workers"] = True
    return kwargs
