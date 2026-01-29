import os
import torch
import torch.distributed as dist

def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="gloo", init_method="env://")
        return rank, local_rank, world_size
    else:
        print("[WARN] Not running in DDP mode. Fallback to single device.")
        if torch.cuda.is_available():
            rank, local_rank, world_size = 0, 0, 1
            torch.cuda.set_device(0)
        else:
            rank, local_rank, world_size = 0, -1, 1
        return rank, local_rank, world_size

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0
