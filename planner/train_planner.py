import argparse
import json
import os
import sys
import time

from tqdm import tqdm
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import imageio.v2 as imageio

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.config_planner import ConfigPlanner
from modules.dataset_lowres import LowResVideoDataset
from modules.transformer import LatentPlanner
from modules.loss import LatentLoss

def setup_ddp():
    import torch.distributed as dist
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="gloo", init_method="env://")
        return rank, local_rank, world_size
    else:
        if torch.cuda.is_available():
            rank, local_rank, world_size = 0, 0, 1
            torch.cuda.set_device(0)
        else:
            rank, local_rank, world_size = 0, -1, 1
        return rank, local_rank, world_size

def cleanup_ddp():
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process():
    import torch.distributed as dist
    return not dist.is_initialized() or dist.get_rank() == 0


def get_dataloader_ddp(config, data_dir, mode="train", rank=0, world_size=1, subset_n=None):
    split = "train" if mode == "train" else "val"
    dataset = LowResVideoDataset(
        root=data_dir,
        split=split,
        target_res=config.target_res,
        max_seq_len=config.max_seq_len,
        )
    
    if subset_n is not None and subset_n > 0:
        dataset.files = dataset.files[:subset_n]
        print(f"[{split}] Using subset: {len(dataset.files)} videos")

    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=(mode == "train"),
        )
    else:
        from torch.utils.data import RandomSampler, SequentialSampler
        sampler = RandomSampler(dataset) if mode == "train" else SequentialSampler(dataset)

    def collate(batch):
        max_len = max(item["length"] for item in batch)
        B = len(batch)
        sample_seq = batch[0]["seq"]  # [T, C, H, W]
        C, H, W = sample_seq.shape[1:]
        
        seq = torch.zeros(B, max_len, C, H, W, dtype=sample_seq.dtype)
        attn = torch.zeros(B, max_len, dtype=batch[0]["attention_mask"].dtype)
        lengths = torch.zeros(B, dtype=torch.long)
        paths = []
        
        for i, item in enumerate(batch):
            l = item["length"]
            seq[i, :l] = item["seq"]
            attn[i, :l] = item["attention_mask"]
            lengths[i] = l
            paths.append(item["path"])
        
        return {"seq": seq, "attention_mask": attn, "lengths": lengths, "paths": paths}

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )


def find_latest_checkpoint(ckpt_dir: str) -> str:
    if not os.path.exists(ckpt_dir):
        return None
    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.startswith("model_epoch_") and f.endswith(".pth")]
    if not ckpt_files:
        return None
    def get_epoch_num(fname):
        try:
            return int(fname.replace("model_epoch_", "").replace(".pth", ""))
        except:
            return -1
    latest = max(ckpt_files, key=get_epoch_num)
    return os.path.join(ckpt_dir, latest)


def load_state_dict(model: torch.nn.Module, ckpt_path: str, device: int) -> None:
    if not ckpt_path:
        return
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=f"cuda:{device}")
    model.load_state_dict(state, strict=False)


@torch.no_grad()
def sample_from_prompt_5(model, prompt_seq: torch.Tensor, total_frames: int, device: torch.device) -> torch.Tensor:
    assert prompt_seq.ndim == 4, f"expected [T,C,H,W], got {prompt_seq.shape}"
    T_avail, C, H, W = prompt_seq.shape
    T_use = min(int(total_frames), int(T_avail))
    assert T_use >= 5, f"need at least 5 frames to prompt, got {T_use}"

    frames = [prompt_seq[i].to(device) for i in range(5)]

    model.eval()
    for _ in range(5, T_use):
        seq = torch.stack(frames, dim=0).unsqueeze(0)
        attn = torch.ones(1, seq.shape[1], device=device)
        pred = model(seq, attn_mask=attn)
        next_frame = pred[:, seq.shape[1]][0]
        frames.append(next_frame)

    return torch.stack(frames, dim=0)


@torch.no_grad()
def save_sample_video(
    model,
    exp_dir,
    epoch,
    sample_seq,
    device,
    max_frames=20,
    cond_frames: int = 5,
    fps: int = 10,
):
    model.eval()
    T_use = min(int(max_frames), int(sample_seq.shape[0]))
    if T_use < cond_frames:
        raise ValueError(f"need at least {cond_frames} frames for visualization, got {T_use}")

    generated = sample_from_prompt_5(
        model=model,
        prompt_seq=sample_seq[:T_use].to(device),
        total_frames=T_use,
        device=device,
    ).cpu()

    gt_seq = sample_seq[: generated.shape[0]].cpu()
    
    def to_img(tensor):
        img = (tensor.clamp(-1, 1) + 1) / 2
        img = img.permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
        return img
    
    frames_per_row = 5
    num_frames = generated.shape[0]
    rows = (num_frames + frames_per_row - 1) // frames_per_row
    
    img_h, img_w = generated.shape[2], generated.shape[3]
    grid_h = rows * img_h * 2
    grid_w = frames_per_row * img_w
    
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    for i in range(num_frames):
        row = i // frames_per_row
        col = i % frames_per_row
        
        gt_img = to_img(gt_seq[i])
        grid[row*img_h:(row+1)*img_h, col*img_w:(col+1)*img_w] = gt_img
        
        gen_img = to_img(generated[i])
        grid[(rows+row)*img_h:(rows+row+1)*img_h, col*img_w:(col+1)*img_w] = gen_img
    
    Image.fromarray(grid).save(os.path.join(exp_dir, f"vis_epoch_{epoch+1}.png"))
    mp4_path = os.path.join(exp_dir, f"vis_epoch_{epoch+1}.mp4")
    try:
        with imageio.get_writer(mp4_path, fps=fps, codec="libx264", quality=8) as w:
            for i in range(num_frames):
                gt_img = to_img(gt_seq[i])
                gen_img = to_img(generated[i])
                frame = np.hstack([gt_img, gen_img])
                w.append_data(frame)
    except Exception as e:
        if is_main_process():
            print(f"[WARN] Failed to save mp4 visualization: {mp4_path} ({e})")

    model.train()


def train(args):
    rank, local_rank, world_size = setup_ddp()

    exp_dir = None
    if is_main_process():
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        exp_name = f"{timestamp}_{args.note}" if args.note else timestamp
        base_exp_dir = getattr(ConfigPlanner, "experiments_root", os.path.join(PROJECT_ROOT, "experiments"))
        os.makedirs(base_exp_dir, exist_ok=True)
        exp_dir = os.path.join(base_exp_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
        print(f"Experiment Directory: {exp_dir}")
        print(f"World Size: {world_size}")

    config = ConfigPlanner()
    data_root = args.data_dir if getattr(args, "data_dir", None) else getattr(config, "data_root", None)
    if not data_root:
        raise ValueError("data_dir must be provided (CLI --data_dir or Config.data_root).")
    config.data_root = data_root
    if args.epochs is not None:
        config.epochs = args.epochs
    
    if getattr(args, "planner_dim", None) is not None:
        config.planner_dim = int(args.planner_dim)
    if getattr(args, "planner_layers", None) is not None:
        config.planner_layers = int(args.planner_layers)
    if getattr(args, "planner_heads", None) is not None:
        config.planner_heads = int(args.planner_heads)
    if getattr(args, "lr", None) is not None:
        config.lr = float(args.lr)
    if getattr(args, "vis_interval", None) is not None:
        config.vis_interval = int(args.vis_interval)
    if getattr(args, "vis_frames", None) is not None:
        config.vis_frames = int(args.vis_frames)
    if getattr(args, "vis_fps", None) is not None:
        config.vis_fps = int(args.vis_fps)

    overfit_n = getattr(args, "overfit_n", None)
    train_subset = getattr(args, "train_subset", None)
    val_subset = getattr(args, "val_subset", None)
    if overfit_n is not None:
        overfit_n = int(overfit_n)
        train_subset = overfit_n if train_subset is None else int(train_subset)
        val_subset = overfit_n if val_subset is None else int(val_subset)

    if is_main_process():
        experiment_summary = {"args": vars(args), "config": config.to_dict()}
        with open(os.path.join(exp_dir, "config.json"), "w") as f:
            json.dump(experiment_summary, f, indent=4)
        loss_tsv_path = os.path.join(exp_dir, "loss_history.tsv")
        if not os.path.exists(loss_tsv_path):
            with open(loss_tsv_path, "w") as f:
                f.write("\t".join(["epoch", "total", "planner", "val_loss", "lr", "grad_norm_clipped"]) + "\n")

    train_loader = get_dataloader_ddp(
        config,
        data_dir=args.data_dir,
        mode="train",
        rank=rank,
        world_size=world_size,
        subset_n=train_subset,
    )
    val_loader = get_dataloader_ddp(
        config,
        data_dir=args.data_dir,
        mode="val",
        rank=rank,
        world_size=world_size,
        subset_n=val_subset,
    )

    # Infer latent_dim from dataset
    sample_seq = train_loader.dataset[0]["seq"]  # [T, C, H, W]
    _, c, h, w = sample_seq.shape
    config.latent_dim = int(c * h * w)
    print(f"Inferred latent_dim: {config.latent_dim} (from {c}x{h}x{w})")

    model = LatentPlanner(config).to(local_rank)
    import torch.distributed as dist
    use_ddp = world_size > 1 and dist.is_initialized()
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    
    def get_model():
        return model.module if use_ddp else model

    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=getattr(config, "weight_decay", 0.01))
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=getattr(config, "min_lr", 0.0))

    loss_fn = LatentLoss(config).to(local_rank)

    # Loss History Memory
    loss_history = {"total": [], "planner": [], "val_loss": []}

    start_epoch = 0
    if args.pretrained:
        if is_main_process():
            print(f"Loading pretrained planner from {args.pretrained}...")
        load_state_dict(get_model(), args.pretrained, local_rank)

    if args.resume:
        if is_main_process():
            print(f"Resuming planner from {args.resume}...")
        load_state_dict(get_model(), args.resume, local_rank)
        opt_path = args.resume.replace("model_epoch_", "optimizer_epoch_")
        sched_path = args.resume.replace("model_epoch_", "scheduler_epoch_")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
        if os.path.exists(sched_path):
            scheduler.load_state_dict(torch.load(sched_path, map_location="cpu"))
        try:
            start_epoch = int(os.path.basename(args.resume).split("_")[-1].split(".")[0])
        except Exception:
            start_epoch = 0

    # Visualization sample
    vis_sample = None
    if is_main_process():
        vis_sample = train_loader.dataset[0]["seq"]  # [T, C, H, W]

    try:
        for epoch in range(start_epoch, config.epochs):
            model.train()
            if hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
            epoch_loss = 0.0
            loss_stats = {"planner": 0.0}
            grad_norm_last = None
            start_time = time.time()

            iterator = train_loader
            if is_main_process():
                iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")

            for batch in iterator:
                seq = batch["seq"].to(local_rank)
                attn_mask = batch["attention_mask"].to(local_rank)

                optimizer.zero_grad(set_to_none=True)
                
                pred_seq = model(seq, attn_mask=attn_mask)
                preds = pred_seq[:, :-1]
                targets = seq
                
                loss, loss_dict = loss_fn(
                    planner_pred=preds,
                    future_latent=targets,
                    eps_pred=None,
                    noise=None,
                    attn_mask=attn_mask,
                )
                loss.backward()
                grad_norm_last = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
                optimizer.step()

                loss_val = float(loss.item())
                epoch_loss += loss_val
                loss_stats["planner"] += float(loss_dict["planner"].item())
                
                if is_main_process():
                    iterator.set_postfix(loss=loss_val)

            val_loss = 0.0
            if is_main_process():
                model.eval()
                val_steps = 0
                with torch.no_grad():
                    for val_batch in val_loader:
                        seq_v = val_batch["seq"].to(local_rank)
                        attn_v = val_batch["attention_mask"].to(local_rank)
                        
                        pred_seq_v = get_model()(seq_v, attn_mask=attn_v)
                        preds_v = pred_seq_v[:, :-1]
                        targets_v = seq_v
                        
                        v_loss, _ = loss_fn(
                            planner_pred=preds_v,
                            future_latent=targets_v,
                            eps_pred=None,
                            noise=None,
                            attn_mask=attn_v,
                        )
                        val_loss += v_loss.item()
                        
                        vis_interval = getattr(config, "vis_interval", 50)
                        vis_frames = getattr(config, "vis_frames", 20)
                        vis_fps = getattr(config, "vis_fps", 10)
                        if val_steps == 0 and vis_sample is not None and ((epoch + 1) % vis_interval == 0 or (epoch + 1) == 1):
                            save_sample_video(
                                get_model(),
                                exp_dir,
                                epoch,
                                vis_sample,
                                local_rank,
                                max_frames=vis_frames,
                                fps=vis_fps,
                            )
                            
                        val_steps += 1
                if val_steps > 0:
                    val_loss /= val_steps

            if is_main_process():
                avg_loss = epoch_loss / len(train_loader)
                loss_stats["planner"] /= len(train_loader)
                
                loss_history["total"].append(avg_loss)
                loss_history["planner"].append(loss_stats["planner"])
                loss_history["val_loss"].append(val_loss)
                
                plt.figure(figsize=(10, 5))
                plt.plot(range(1, len(loss_history["total"]) + 1), loss_history["total"], label="Train Loss")
                plt.plot(range(1, len(loss_history["val_loss"]) + 1), loss_history["val_loss"], label="Val Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title("Training Loss Curve")
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(exp_dir, "loss_plot.png"))
                plt.close()

                print(
                    f"Epoch [{epoch+1}/{config.epochs}] "
                    f"Loss: {avg_loss:.4f} Val: {val_loss:.4f} "
                    f"(Planner: {loss_stats['planner']:.4f}) "
                    f"Time: {time.time() - start_time:.1f}s"
                )

                with open(os.path.join(exp_dir, "loss_history.tsv"), "a") as f:
                    lr = optimizer.param_groups[0].get("lr", None)
                    f.write("\t".join([
                        str(epoch + 1),
                        f"{avg_loss:.6f}",
                        f"{loss_stats['planner']:.6f}",
                        f"{val_loss:.6f}",
                        f"{lr:.8f}" if lr is not None else "NA",
                        f"{float(grad_norm_last):.6f}" if grad_norm_last is not None else "NA",
                    ]) + "\n")

                if (epoch + 1) % config.ckpt_interval == 0:
                    ckpt_path = os.path.join(exp_dir, "checkpoints", f"model_epoch_{epoch+1}.pth")
                    torch.save(get_model().state_dict(), ckpt_path)
                    opt_path2 = os.path.join(exp_dir, "checkpoints", f"optimizer_epoch_{epoch+1}.pth")
                    torch.save(optimizer.state_dict(), opt_path2)
                    sched_path2 = os.path.join(exp_dir, "checkpoints", f"scheduler_epoch_{epoch+1}.pth")
                    torch.save(scheduler.state_dict(), sched_path2)

            scheduler.step()
    finally:
        cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--note", type=str, default="", help="Description of the experiment")
    parser.add_argument("--data_dir", type=str, default=ConfigPlanner.data_root, help="Path to processed dataset root")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained weights to load")
    parser.add_argument("--epochs", type=int, default=None, help="Override ConfigPlanner.epochs")
    parser.add_argument("--vis_interval", type=int, default=None, help="Override ConfigPlanner.vis_interval")
    parser.add_argument("--vis_frames", type=int, default=None, help="Override ConfigPlanner.vis_frames")
    parser.add_argument("--vis_fps", type=int, default=None, help="Override ConfigPlanner.vis_fps")
    parser.add_argument("--overfit_n", type=int, default=None, help="Limit both train/val to first N samples")
    parser.add_argument("--train_subset", type=int, default=None, help="Limit train set to first N samples")
    parser.add_argument("--val_subset", type=int, default=None, help="Limit val set to first N samples")
    parser.add_argument("--planner_dim", type=int, default=None, help="Override planner_dim")
    parser.add_argument("--planner_layers", type=int, default=None, help="Override planner_layers")
    parser.add_argument("--planner_heads", type=int, default=None, help="Override planner_heads")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    args = parser.parse_args()

    train(args)
