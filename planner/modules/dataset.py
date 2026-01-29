import os
import random
from typing import List, Tuple

import torch
from torch.utils.data import Dataset


class LatentSequenceDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        max_seq_len: int,
        random_crop: Tuple[int, int] | None = None,
    ):
        self.root = root
        self.split = split
        self.max_seq_len = int(max_seq_len)
        self.random_crop = random_crop

        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"split dir not found: {split_dir}")

        files = [f for f in os.listdir(split_dir) if f.lower().endswith(".pt")]
        files.sort()
        self.paths: List[str] = [os.path.join(split_dir, f) for f in files]
        if len(self.paths) == 0:
            raise RuntimeError(f"no .pt files found in {split_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        path = self.paths[idx]
        obj = torch.load(path, map_location="cpu")
        if "z" not in obj:
            raise KeyError(f"{path}: missing 'z' key (expected encode.py output)")
        z = obj["z"]
        if not torch.is_tensor(z) or z.ndim not in (2, 4):
            raise ValueError(f"{path}: 'z' must be [T, D] or [T, C, H, W]")

        T = z.shape[0]

        if self.random_crop is not None:
            l_min, l_max = self.random_crop
            l_min = max(1, int(l_min))
            l_max = max(l_min, int(l_max))
            target_len = random.randint(l_min, min(l_max, T))
        else:
            target_len = T

        target_len = min(target_len, self.max_seq_len)
        z = z[:target_len]
        if z.ndim == 2:
            seq = z.unsqueeze(1)  # [T', 1, D]
        else:
            seq = z  # [T', C, H, W]
        attn_mask = torch.ones(seq.shape[0], dtype=torch.float32)  # [T']

        return {
            "seq": seq,
            "attention_mask": attn_mask,
            "length": seq.shape[0],
            "path": path,
        }
