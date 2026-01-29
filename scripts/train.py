import argparse
import os
import glob
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import random
import time
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from utils_ddp import setup_ddp, cleanup_ddp, is_main_process

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.diffusion import AsymmetricUNet
from models.scheduler import NoiseScheduler

def parse_args():
    parser = argparse.ArgumentParser(description="Train Asymmetric Diffusion (Progressive Growth)")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to PROCESSED data root")
    parser.add_argument("--high_res", type=int, default=128, help="Original Size (H, W)")
    parser.add_argument("--latent_res", type=int, default=32, help="Latent Size (Minimum Resolution)")
    parser.add_argument("--k_step", type=int, default=2, help="Resolution Step K")
    
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    
    parser.add_argument("--save_interval", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--vis_interval", type=int, default=1, help="Visualize every N epochs")
    parser.add_argument("--exp_name", type=str, default="default", help="Experiment name")
    parser.add_argument("--num_workers", type=int, default=8)
    
    # Model Architecture Options
    parser.add_argument("--model_channels", type=int, default=128, help="Base channel count")
    parser.add_argument("--channel_mult", type=int, nargs="+", default=[1, 2, 2, 2], help="Channel multipliers per level")
    parser.add_argument("--num_res_blocks", type=int, default=2, help="ResBlocks per level")
    parser.add_argument("--subset_n", type=int, default=None, help="Use only first N videos for training")
    parser.add_argument("--obj_weight", type=float, default=100.0, help="Weight for object pixels")

    args = parser.parse_args()
    return args

def get_resolution_for_timestep(t, num_timesteps, high_res, latent_res, k_step):
    ratio = t / (num_timesteps - 1)
    size_float = high_res - ratio * (high_res - latent_res)
    size_int = int(round(size_float / k_step) * k_step)
    size_int = max(latent_res, min(high_res, size_int))
    return size_int
class TensorVideoDataset(Dataset):
    def __init__(self, root, split="train", frames_per_video=100, subset_n=None):
        self.root = root
        self.split = split
        self.frames_per_video = frames_per_video
        
        split_dir = os.path.join(root, split)
        if not os.path.exists(split_dir):
            split_dir = root
            
        self.files = sorted(glob.glob(os.path.join(split_dir, "*.pt")))
        
        if subset_n is not None and subset_n > 0:
            print(f"[{split}] Subsetting dataset to first {subset_n} videos.")
            self.files = self.files[:subset_n]
            
        print(f"[{split}] Found {len(self.files)} processed video files.")
        
    def __len__(self):
        return len(self.files) * self.frames_per_video

    def __getitem__(self, idx):
        video_idx = idx // self.frames_per_video
        frame_idx = idx % self.frames_per_video
            
        fpath = self.files[video_idx]
        video_tensor = torch.load(fpath, map_location="cpu") # [T, 3, H, W]
        frame = video_tensor[frame_idx] # [3, H, W]
        frame = frame.float() / 255.0
        frame = (frame - 0.5) * 2.0
        return frame

@torch.no_grad()
def save_sample_images(model, scheduler, device, exp_dir, epoch, latent_res, high_res, k_step, sample_input=None):
    model.eval()
    if sample_input is None: sample_input = torch.randn(1, 3, high_res, high_res).to(device)
    
    gt_img = sample_input.to(device)
    
    curr_res = latent_res
    curr_sample = torch.randn(1, 3, curr_res, curr_res).to(device)
    
    t_start = int((scheduler.num_timesteps - 1) * 0.1)
    
    latent_img = F.interpolate(gt_img, size=(latent_res, latent_res), mode='bilinear', align_corners=False)
    
    noise = torch.randn_like(latent_img)
    timesteps = torch.tensor([t_start], device=device).long()
    curr_sample = scheduler.add_noise(latent_img, noise, timesteps)
    
    noisy_input_view = curr_sample.clone()
    if noisy_input_view.shape[-1] != high_res:
        noisy_input_view = F.interpolate(noisy_input_view, size=(high_res, high_res), mode='nearest')
    
    for t in tqdm(reversed(range(0, t_start + 1)), desc="Sampling", leave=False):
        t_batch = torch.tensor([t], device=device).long()
        
        target_res_t = get_resolution_for_timestep(t, scheduler.num_timesteps, high_res, latent_res, k_step)
        
        if curr_sample.shape[-1] != target_res_t:
             curr_sample = F.interpolate(curr_sample, size=(target_res_t, target_res_t), mode='bilinear', align_corners=False)
        
        pred_x0_high = model(curr_sample, t_batch, target_shape=(high_res, high_res))
        
        if t > 0:
            next_res = get_resolution_for_timestep(t-1, scheduler.num_timesteps, high_res, latent_res, k_step)
        else:
            next_res = high_res
            
        pred_x0_curr = F.interpolate(pred_x0_high, size=(target_res_t, target_res_t), mode='bilinear', align_corners=False)
        
        prev_sample = scheduler.step_x0(pred_x0_curr, t_batch, curr_sample)
        
        if t > 0 and prev_sample.shape[-1] != next_res:
             curr_sample = F.interpolate(prev_sample, size=(next_res, next_res), mode='bilinear', align_corners=False)
        else:
             curr_sample = prev_sample
        
    res_img_tensor = curr_sample
    if res_img_tensor.shape[-1] != high_res:
        res_img_tensor = F.interpolate(res_img_tensor, size=(high_res, high_res), mode='bilinear', align_corners=False)
        
    def to_img(tensor):
        img = (tensor.clamp(-1, 1) + 1) / 2
        img = img.permute(0, 2, 3, 1).cpu().numpy()
        img = (img * 255).astype(np.uint8)[0]
        return img
    
    res_img = to_img(res_img_tensor)
    gt_img_np = to_img(gt_img)
    latent_view = to_img(F.interpolate(latent_img, size=(high_res, high_res), mode='nearest'))
    noisy_view_np = to_img(noisy_input_view)
    
    combined = np.hstack([latent_view, noisy_view_np, res_img, gt_img_np])
    Image.fromarray(combined).save(os.path.join(exp_dir, f"vis_epoch_{epoch+1}.png"))
    model.train()

def train(args):
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main_process():
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        exp_dir = os.path.join("experiments", f"{timestamp}_{args.exp_name}")
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
        print(f"Experimental Dir: {exp_dir}")
        with open(os.path.join(exp_dir, "args.txt"), "w") as f: f.write(str(vars(args)))
        
        # Initialize Loss History
        loss_history = []
        with open(os.path.join(exp_dir, "loss_history.tsv"), "w") as f:
            f.write("epoch\tloss\n")

    # Resolution Settings
    HIGH_RES = args.high_res
    LATENT_RES = args.latent_res
    K_STEP = args.k_step
    
    if is_main_process():
        print(f"Progressive Growth: Latent {LATENT_RES} <--(t)--> High {HIGH_RES} (Step K={K_STEP})")
        print(f"Model: Channels={args.model_channels}, Mult={args.channel_mult}, Blocks={args.num_res_blocks}")

    model = AsymmetricUNet(
        in_channels=3, 
        out_channels=3, 
        model_channels=args.model_channels, 
        channel_mult=tuple(args.channel_mult), 
        num_res_blocks=args.num_res_blocks
    ).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        model_module = model.module
    else:
        model_module = model

    scheduler = NoiseScheduler(device=device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    frames_per_video = 100
    dataset = TensorVideoDataset(args.data_dir, split="train", frames_per_video=frames_per_video, subset_n=args.subset_n)
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    start_epoch = 0
    vis_sample = None
    if is_main_process():
        val_set = TensorVideoDataset(args.data_dir, split="val", frames_per_video=frames_per_video)
        if len(val_set) > 0: vis_sample = val_set[0].unsqueeze(0).to(device)
        elif len(dataset) > 0: vis_sample = dataset[0].unsqueeze(0).to(device)
        print("Visualization sample loaded.")
        
        # Initial Visualization Test
        if vis_sample is not None:
            print("Running initial visualization test...")
            save_sample_images(model_module, scheduler, device, exp_dir, -1, LATENT_RES, HIGH_RES, K_STEP, sample_input=vis_sample)
            print("Initial visualization saved.")

    for epoch in range(start_epoch, args.epochs):
        if sampler: sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        
        iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}") if is_main_process() else dataloader
            
        for batch_idx, high_res_images in enumerate(iterator):
            high_res_images = high_res_images.to(device)
            B = high_res_images.shape[0]
            
            t_scalar = random.randint(0, scheduler.num_timesteps - 1)
            t_batch = torch.full((B,), t_scalar, device=device).long()
            
            current_res = get_resolution_for_timestep(t_scalar, scheduler.num_timesteps, HIGH_RES, LATENT_RES, K_STEP)
            current_shape = (current_res, current_res)
            
            noise = torch.randn_like(high_res_images)
            noisy_high_res = scheduler.add_noise(high_res_images, noise, t_batch)
            
            low_res_input = F.interpolate(noisy_high_res, size=current_shape, mode='bilinear', align_corners=False)
            
            pred_x0 = model(low_res_input, t_batch, target_shape=(HIGH_RES, HIGH_RES))
            
            intensity = (high_res_images + 1) * 0.5
            brightness = intensity.mean(dim=1, keepdim=True)
            
            pixel_weight = 1.0 + args.obj_weight * brightness
            
            diff = torch.abs(pred_x0 - high_res_images)
            loss = (diff * pixel_weight).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            if is_main_process(): iterator.set_postfix(loss=loss.item(), res=current_res)

        if is_main_process():
            avg_loss = epoch_loss / len(dataloader) if len(dataloader) > 0 else 0
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1} Done. Avg Loss: {avg_loss:.4f} | LR: {current_lr:.8f}")
            
            loss_history.append(avg_loss)
            with open(os.path.join(exp_dir, "loss_history.tsv"), "a") as f:
                f.write(f"{epoch+1}\t{avg_loss:.6f}\t{current_lr:.8f}\n")
                
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, len(loss_history) + 1), loss_history, label="Train Loss")
            plt.xlabel("Epoch")
            plt.ylabel("MSE Loss")
            plt.title("Training Loss Curve")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(exp_dir, "loss_plot.png"))
            plt.close()
            
            if (epoch + 1) % args.vis_interval == 0 and vis_sample is not None:
                save_sample_images(model_module, scheduler, device, exp_dir, epoch, LATENT_RES, HIGH_RES, K_STEP, sample_input=vis_sample)
            
            if (epoch + 1) % args.save_interval == 0:
                ckpt_path = os.path.join(exp_dir, "checkpoints", f"model_epoch_{epoch+1}.pt")
                torch.save(model_module.state_dict(), ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")
        
        if lr_scheduler is not None:
            lr_scheduler.step()

    cleanup_ddp()

if __name__ == "__main__":
    args = parse_args()
    train(args)
