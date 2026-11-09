import copy

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from sirentv.utils.registry import Registry

DATASETS = Registry("datasets")


def _register_all_datasets():
    """Import dataset modules so they register with DATASETS."""
    import sirentv.data.compressed  # noqa: F401 — CompressedPLibDataset
    import sirentv.data.quantile  # noqa: F401 — QuantilePLibDataset


def build_dataset(cfg, rank=0, world_size=1):
    """Build cfg["data"]["dataset"] (a registry type) with the full cfg, rank and world_size."""
    _register_all_datasets()
    dataset_cfg = copy.deepcopy(cfg["data"]["dataset"])
    default_args = dict(cfg=cfg, rank=rank, world_size=world_size)
    return DATASETS.build(dataset_cfg, default_args=default_args)


def create_dataloader(cfg, rank=0, world_size=1):
    """Build the dataset and wrap it per cfg["data"]["loader"]."""
    dataset = build_dataset(cfg, rank=rank, world_size=world_size)

    loader_cfg = cfg["data"].get("loader", {})
    batch_size = loader_cfg.get("batch_size", 1)
    num_workers = loader_cfg.get("num_workers", 0)
    pin_memory = loader_cfg.get("pin_memory", False)
    drop_last = loader_cfg.get("drop_last", False)
    shuffle = loader_cfg.get("shuffle", True)
    prefetch_factor = loader_cfg.get("prefetch_factor")

    # a memory shard (QuantilePLibDataset data.memory_shard) is already split across ranks
    if getattr(dataset, "_memory_shard", False):
        from sirentv.data.quantile import MemoryShardLoader

        return MemoryShardLoader(
            dataset, batch_size, shuffle=shuffle, drop_last=drop_last,
            seed=cfg.get("train", {}).get("seed") or 0, rank=rank, world_size=world_size,
        )

    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
        )
        shuffle = False  # sampler handles shuffling

    # workers cannot share CUDA tensors
    if getattr(dataset, "_on_gpu", False):
        num_workers = 0
        pin_memory = False

    extra = {}
    if num_workers > 0 and prefetch_factor is not None:  # DataLoader rejects it with 0 workers
        extra["prefetch_factor"] = int(prefetch_factor)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        sampler=sampler,
        persistent_workers=num_workers > 0,
        **extra,
    )
