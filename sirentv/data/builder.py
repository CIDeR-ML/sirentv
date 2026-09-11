import copy

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from sirentv.utils.registry import Registry

DATASETS = Registry("datasets")


def _register_all_datasets():
    """Import dataset modules so they register with the DATASETS registry."""
    import sirentv.data.io  # noqa: F401 — registers PLibDataset
    import sirentv.data.compressed  # noqa: F401 — registers CompressedPLibDataset
    import sirentv.data.quantile  # noqa: F401 — registers QuantilePLibDataset


def build_dataset(cfg, rank=0, world_size=1):
    """Build a dataset from config.

    Args:
        cfg (dict): Full training config. The dataset sub-config is read
            from ``cfg["data"]["dataset"]``.
        rank (int): Process rank for distributed training.
        world_size (int): Total number of processes.

    Returns:
        Dataset: The constructed dataset.
    """
    _register_all_datasets()
    dataset_cfg = copy.deepcopy(cfg["data"]["dataset"])
    default_args = dict(cfg=cfg, rank=rank, world_size=world_size)
    return DATASETS.build(dataset_cfg, default_args=default_args)


def create_dataloader(cfg, rank=0, world_size=1):
    """Build a dataset and wrap it in a DataLoader.

    Args:
        cfg (dict): Full training config. Dataset config is read from
            ``cfg["data"]["dataset"]`` and loader config from
            ``cfg["data"]["loader"]``.
        rank (int): Process rank for distributed training.
        world_size (int): Total number of processes.

    Returns:
        DataLoader: The constructed data loader.
    """
    dataset = build_dataset(cfg, rank=rank, world_size=world_size)

    loader_cfg = cfg["data"].get("loader", {})
    batch_size = loader_cfg.get("batch_size", 1)
    num_workers = loader_cfg.get("num_workers", 0)
    pin_memory = loader_cfg.get("pin_memory", False)
    drop_last = loader_cfg.get("drop_last", False)
    shuffle = loader_cfg.get("shuffle", True)

    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
        )
        shuffle = False  # sampler handles shuffling

    # If the dataset lives on GPU, DataLoader workers cannot share CUDA
    # tensors, so force single-process loading with no pinning.
    if getattr(dataset, "_on_gpu", False):
        num_workers = 0
        pin_memory = False

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        sampler=sampler,
        persistent_workers=num_workers > 0,
    )

    return dl
