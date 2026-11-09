import os

import torch.distributed as comm
from torch.nn.parallel import DistributedDataParallel


def create_ddp_model(model, *, fp16_compression=False, **kwargs):
    """Wrap `model` in DistributedDataParallel when there is more than one process.

    fp16_compression adds DDP's fp16_compress_hook; kwargs go to DistributedDataParallel.
    """
    if not comm.is_initialized() or comm.get_world_size() == 1:
        return model

    if "device_ids" not in kwargs:
        kwargs["device_ids"] = [int(os.environ.get('LOCAL_RANK', 0))]
    ddp = DistributedDataParallel(model, **kwargs)

    if fp16_compression:
        from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks
        ddp.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
    return ddp
