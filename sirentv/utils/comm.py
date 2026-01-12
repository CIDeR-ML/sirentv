from torch.nn.parallel import DistributedDataParallel
import torch.distributed as comm
import os

def create_ddp_model(model, *, fp16_compression=False, **kwargs):
    """
    Create a DistributedDataParallel model if there are >1 processes.
    Args:
        model: a torch.nn.Module
        fp16_compression: add fp16 compression hooks to the ddp object.
            See more at https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook
        kwargs: other arguments of :module:`torch.nn.parallel.DistributedDataParallel`.
    """
    if not comm.is_initialized() or comm.get_world_size() == 1:
        return model

    # Get local rank from environment variable
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    # kwargs['find_unused_parameters'] = True
    if "device_ids" not in kwargs:
        kwargs["device_ids"] = [local_rank]
        # https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
        # For multi-device modules and CPU modules, it must be None
        #if "output_device" not in kwargs:
        #    kwargs["output_device"] = [local_rank]
    ddp = DistributedDataParallel(model, **kwargs)

    if fp16_compression:
        from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks
        ddp.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
    return ddp
