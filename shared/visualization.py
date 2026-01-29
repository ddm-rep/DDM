
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import imageio
from vae.vae import FrameVAE, FrameVAEConfig

def load_vae_model(ckpt_path, device):
    """
    Load FrameVAE from checkpoint
    """
    if not os.path.exists(ckpt_path):
        print(f"[WARN] VAE checkpoint not found: {ckpt_path}")
        return None

    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    # Load Config
    cfg_dict = ckpt.get("cfg", None)
    if cfg_dict is None:
        # Try to infer from model state or use default
        print("[WARN] No config in checkpoint, using default FrameVAEConfig")
        cfg = FrameVAEConfig() 
    else:
        # If cfg_dict is a FrameVAEConfig object or dict
        if isinstance(cfg_dict, dict):
            cfg = FrameVAEConfig(**cfg_dict)
        else:
            cfg = cfg_dict

    model = FrameVAE(cfg).to(device)
    
    # Load State Dict
    state = ckpt.get("model", ckpt.get("model_state_dict", None))
    if state is None:
        # Try top level
        state = ckpt
        
    # Handle DDP prefix
    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            new_state[k[7:]] = v
        else:
            new_state[k] = v
            
    model.load_state_dict(new_state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def save_comparison_video(
    pred_latents: torch.Tensor,
    gt_latents: torch.Tensor,
    vae: torch.nn.Module,
    output_path: str,
    fps: int = 10,
    attn_mask: torch.Tensor | None = None
):
    """
    pred_latents, gt_latents: [T, C, H, W]
    Saves side-by-side comparison video using VAE decoder.
    """
    device = pred_latents.device
    
    # Ensure lengths match
    min_len = min(pred_latents.shape[0], gt_latents.shape[0])
    pred_latents = pred_latents[:min_len]
    gt_latents = gt_latents[:min_len]
    
    if attn_mask is not None:
        valid_indices = torch.where(attn_mask[:min_len].bool())[0]
        pred_latents = pred_latents[valid_indices]
        gt_latents = gt_latents[valid_indices]
        min_len = len(valid_indices)

    T = min_len
    if T == 0:
        return
        
    frames = []
    batch_size = 8 # Depends on GPU memory
    
    for i in range(0, T, batch_size):
        batch_pred = pred_latents[i : i + batch_size]
        batch_gt = gt_latents[i : i + batch_size]
        
        # Decode using VAE.decode() -> [B, 3, H, W]
        recon_pred = vae.decode(batch_pred).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
        recon_gt = vae.decode(batch_gt).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
        
        # [0,1] -> [0,255]
        recon_pred = (recon_pred * 255).astype(np.uint8)
        recon_gt = (recon_gt * 255).astype(np.uint8)
        
        # Concatenate side-by-side [GT | Pred]
        for j in range(recon_pred.shape[0]):
            combined = np.concatenate([recon_gt[j], recon_pred[j]], axis=1)
            frames.append(combined)
            
    if len(frames) == 0:
        return

    try:
        imageio.mimsave(output_path, frames, fps=fps, codec='libx264', quality=8)
    except Exception as e:
        print(f"[ERROR] Failed to save video: {e}")
        try:
             imageio.mimsave(output_path, frames, fps=fps)
        except Exception:
             pass

def plot_loss(loss_history, output_path):
    """
    loss_history: dict of lists
      - required: {"total": [], "planner": [], "diffusion": []}
    """
    epochs = range(1, len(loss_history["total"]) + 1)
    
    # Check what keys are available
    keys = list(loss_history.keys())
    
    fig, axes = plt.subplots(1, 1, figsize=(10, 5))
    
    # 1. Planner Loss
    if "planner" in loss_history:
        axes.plot(epochs, loss_history["planner"], label='Train Planner', color='blue')
        
    # Validation Loss
    if "val_loss" in loss_history and len(loss_history["val_loss"]) == len(epochs):
        axes.plot(epochs, loss_history["val_loss"], label='Val Loss', color='orange', linestyle='--')
        
    axes.set_title('Training Loss')
    axes.legend()
    axes.set_xlabel('Epoch')
    axes.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
