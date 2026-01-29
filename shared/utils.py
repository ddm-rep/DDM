import torch
import torch.distributed as dist
import os
from torch.utils.data import Subset

def set_requires_grad(module: torch.nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad = requires_grad

def setup_ddp():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

    torch.cuda.set_device(local_rank)
    backend = 'nccl' if world_size > 1 else 'gloo'
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return rank, local_rank, world_size

def cleanup_ddp():
    dist.destroy_process_group()

def N_subset_dataset(dataset, subset_n: int | None):
    if subset_n <= 0:
        raise ValueError(f"subset_n must be positive, got {subset_n}")
    n = min(int(subset_n), len(dataset))
    # Deterministic subset for sanity checks (first N samples).
    return Subset(dataset, list(range(n)))

def rank0_print(message: str):
    if dist.get_rank() == 0:
        print(message)