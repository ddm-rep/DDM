#!/usr/bin/env python
import argparse
import glob
import os
from functools import partial
from multiprocessing import Pool

import torch
import torch.nn.functional as F
from tqdm import tqdm


def _convert_one(src_path: str, dst_path: str, target_res: int) -> bool:
    """
    Convert one processed video tensor to low-res tensor.
    Expected input format:
      - torch.Tensor uint8 [T,3,H,W]  (what scripts/preprocess_data.py writes), OR
      - dict with key "frames" uint8 [T,3,H,W]
    Output:
      - torch.Tensor uint8 [T,3,target_res,target_res]
    """
    obj = torch.load(src_path, map_location="cpu")
    if isinstance(obj, dict) and "frames" in obj:
        frames = obj["frames"]
    elif torch.is_tensor(obj):
        frames = obj
    else:
        raise ValueError(f"{src_path}: unsupported format (expected Tensor or dict with 'frames')")

    if frames.ndim != 4 or int(frames.shape[1]) != 3:
        raise ValueError(f"{src_path}: expected [T,3,H,W], got {tuple(frames.shape)}")

    # uint8 -> float for resize
    x = frames.to(torch.float32)
    x = F.interpolate(x, size=(int(target_res), int(target_res)), mode="bilinear", align_corners=False)
    x = x.clamp(0, 255).to(torch.uint8)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    torch.save(x, dst_path)
    return True


def _worker(task: tuple[str, str], target_res: int) -> bool:
    src, dst = task
    if os.path.exists(dst):
        return True
    return _convert_one(src, dst, target_res)


def main():
    p = argparse.ArgumentParser(description="Prepare low-res planner dataset from processed tensors")
    p.add_argument("--in_dir", type=str, default="/workspace/data/processed", help="Input root (train/val/test/*.pt)")
    p.add_argument("--out_dir", type=str, default="/workspace/data/processed_32", help="Output root")
    p.add_argument("--target_res", type=int, default=32, help="Low-res size")
    p.add_argument("--workers", type=int, default=8, help="Multiprocessing workers")
    p.add_argument("--limit", type=int, default=0, help="Limit number of videos per split (0 = no limit)")
    args = p.parse_args()

    for split in ("train", "val", "test"):
        src_dir = os.path.join(args.in_dir, split)
        if not os.path.isdir(src_dir):
            continue
        dst_dir = os.path.join(args.out_dir, split)
        os.makedirs(dst_dir, exist_ok=True)

        src_files = sorted(glob.glob(os.path.join(src_dir, "*.pt")))
        if args.limit and int(args.limit) > 0:
            src_files = src_files[: int(args.limit)]

        tasks: list[tuple[str, str]] = []
        for src in src_files:
            name = os.path.basename(src)
            dst = os.path.join(dst_dir, name)
            tasks.append((src, dst))

        print(f"[{split}] {len(tasks)} files -> {dst_dir} (target_res={args.target_res})")
        if len(tasks) == 0:
            continue

        func = partial(_worker, target_res=int(args.target_res))
        with Pool(int(args.workers)) as pool:
            for _ in tqdm(pool.imap_unordered(func, tasks), total=len(tasks)):
                pass


if __name__ == "__main__":
    main()

