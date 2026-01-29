#!/usr/bin/env python
import argparse
import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from vae.model import FrameEncoder, VAEConfig
from vae.video_io import read_video_rgb

# Simple Dataset class to replace the missing VideoFrameDataset
class RawVideoDataset(Dataset):
    def __init__(self, root, split, frame_size=None, limit=None):
        self.root = root
        self.split = split
        self.frame_size = frame_size
        
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
            
        self.files = sorted(glob.glob(os.path.join(split_dir, "*.mp4")))
        if limit:
            self.files = self.files[:limit]
            
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        path = self.files[idx]
        # read_video_rgb returns [T, H, W, 3] uint8
        # We need to handle resizing if frame_size is provided, 
        # but for simplicity let's assume VAE handles resizing or input is correct.
        # Ideally, resize here.
        vid = read_video_rgb(path) 
        
        # Resize if needed (simple approximation using torch.nn.functional.interpolate)
        vid_tensor = torch.from_numpy(vid).permute(0, 3, 1, 2).float() / 255.0 # [T, 3, H, W]
        
        if self.frame_size:
             vid_tensor = torch.nn.functional.interpolate(
                 vid_tensor, 
                 size=(self.frame_size, self.frame_size), 
                 mode='bilinear', 
                 align_corners=False
             )

        return {"video": vid_tensor, "path": path}

def parse_args():
    p = argparse.ArgumentParser(description="VAE latent sequence generator")
    p.add_argument("--video-root", type=str, required=True, help="Source video root (train/val/test subdirectories with mp4)")
    p.add_argument("--out-dir", type=str, required=True, help="Output directory (.pt files)")
    p.add_argument("--ckpt", type=str, required=True, help="VAE checkpoint path")
    p.add_argument("--split", type=str, default="train", help="Split to process (train|val|test)")
    p.add_argument("--batch-size", type=int, default=32, help="Encoding batch size")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--latent-dim", type=int, default=1024, help="Config override")
    p.add_argument("--base-channels", type=int, default=128, help="Config override")
    p.add_argument("--frame-size", type=int, default=128, help="Resize target")
    p.add_argument("--use-mu", action="store_true", help="Use mean only (deterministic)")
    p.add_argument("--limit", type=int, default=0, help="Limit number of videos")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading VAE from {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    
    # Load Config from checkpoint or args
    cfg_dict = ckpt.get("cfg", None)
    if cfg_dict:
        cfg = VAEConfig(**cfg_dict)
    else:
        cfg = VAEConfig(
            latent_dim=args.latent_dim,
            base_channels=args.base_channels,
            frame_size=args.frame_size
        )
    
    # Override config with args if explicitly provided and different? 
    # For now, trust checkpoint config if exists, otherwise args.
    
    enc = FrameEncoder(
        in_channels=cfg.in_channels,
        latent_dim=cfg.latent_dim,
        base_channels=cfg.base_channels
    ).to(device)
    
    state = ckpt.get("encoder") or ckpt.get("model")
    enc.load_state_dict(state, strict=False)
    enc.eval()
    
    os.makedirs(args.out_dir, exist_ok=True)
    out_split_dir = os.path.join(args.out_dir, args.split)
    os.makedirs(out_split_dir, exist_ok=True)
    
    dataset = RawVideoDataset(args.video_root, args.split, frame_size=cfg.frame_size, limit=args.limit)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers) # Process one video at a time
    
    print(f"Processing {len(dataset)} videos from {args.split}...")
    
    with torch.no_grad():
        for item in tqdm(loader):
            # item["video"] is [1, T, 3, H, W] due to batch_size=1
            vid = item["video"][0] 
            path = item["path"][0]
            
            T = vid.shape[0]
            z_list, mu_list, logvar_list = [], [], []
            
            # Batch processing frames within a video
            for i in range(0, T, args.batch_size):
                batch = vid[i : i + args.batch_size].to(device)
                mu, logvar = enc(batch)
                
                if args.use_mu:
                    z = mu
                else:
                    std = torch.exp(0.5 * logvar)
                    z = mu + torch.randn_like(std) * std
                    
                z_list.append(z.cpu())
                mu_list.append(mu.cpu())
                logvar_list.append(logvar.cpu())
            
            z_seq = torch.cat(z_list, dim=0)
            mu_seq = torch.cat(mu_list, dim=0)
            logvar_seq = torch.cat(logvar_list, dim=0)
            
            name = os.path.splitext(os.path.basename(path))[0]
            save_path = os.path.join(out_split_dir, f"{name}.pt")
            
            torch.save({
                "z": z_seq, # [T, D]
                "mu": mu_seq,
                "logvar": logvar_seq,
                "src": path
            }, save_path)
            
    print("Done.")

if __name__ == "__main__":
    main()
