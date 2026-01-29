import os
import glob
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class LowResVideoDataset(Dataset):
    
    def __init__(
        self,
        root: str,
        split: str,
        target_res: int = 32,
        max_seq_len: int = 100,
    ):
        self.root = root
        self.split = split
        self.target_res = target_res
        self.max_seq_len = max_seq_len
        
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"split dir not found: {split_dir}")
        
        self.files = sorted(glob.glob(os.path.join(split_dir, "*.pt")))
        if len(self.files) == 0:
            raise RuntimeError(f"no .pt files found in {split_dir}")
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> dict:
        path = self.files[idx]
        obj = torch.load(path, map_location="cpu")
        
        # Load frames: [T, 3, H, W] (uint8)
        if isinstance(obj, dict) and "frames" in obj:
            frames = obj["frames"]
        elif isinstance(obj, torch.Tensor):
            frames = obj
        else:
            raise ValueError(f"{path}: unexpected format")
        
        T, C, H, W = frames.shape
        if C != 3:
            raise ValueError(f"{path}: expected 3 channels, got {C}")
        
        frames = frames.float() / 255.0
        frames = (frames - 0.5) * 2.0
        
        if H != self.target_res or W != self.target_res:
            raise ValueError(
                f"{path}: expected preprocessed frames at {self.target_res}x{self.target_res}, got {H}x{W}. "
                f"Run: python planner/prepare_lowres.py --in_dir /workspace/data/processed "
                f"--out_dir /workspace/data/processed_32 --target_res {self.target_res}"
            )
        
        T = min(T, self.max_seq_len)
        frames = frames[:T]
        
        attn_mask = torch.ones(T, dtype=torch.float32)
        
        return {
            "seq": frames,
            "attention_mask": attn_mask,
            "length": T,
            "path": path,
        }
