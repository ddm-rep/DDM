import os


class Config:
    # --------------------------------------------------------------------------
    # 1. Data & Path
    # --------------------------------------------------------------------------
    data_root = "/workspace/data/vae_latents"
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    experiments_root = os.path.join(project_root, "experiments")
    vae_latent_root = os.path.join(experiments_root, "vae_latents")
    vae_recon_root = os.path.join(experiments_root, "vae_recon")

    # DataLoader
    batch_size = 64
    num_workers = 8
    
    # Sequence Processing
    max_seq_len = 200        # Max length for padding/positional encoding
    random_crop = (40, 100)  # (min, max) length for random cropping during training
    
    # --------------------------------------------------------------------------
    # 2. Model Architecture
    # --------------------------------------------------------------------------
    # VAE (Fixed, Pre-trained)
    latent_dim = 1024        # Must match the VAE checkpoint
    base_channels = 128
    vae_ckpt_path = "/workspace/all_together/vae/vae_weight/checkpoints/best.pth" # Path to VAE checkpoint for visualization
    
    # Planner (LLM - Autoregressive)
    planner_dim = 1024
    planner_heads = 16
    planner_layers = 12
    planner_dropout = 0.1
    planner_temperature = 1.0
    
    # Refiner (Diffusion - Bidirectional)
    diffusion_dim = 1024
    diffusion_heads = 16
    diffusion_layers = 12
    diffusion_dropout = 0.1
    
    diffusion_steps = 10 
    beta_start = 0.0001
    beta_end = 0.02
    diffusion_truncation = 0.5  # Ratio of steps to use (0.0 < r <= 1.0)
    
    # Compatibility (Internal use)
    state_dim = latent_dim
    max_objects = 1

    # --------------------------------------------------------------------------
    # 3. Training Configuration
    # --------------------------------------------------------------------------
    stage = "refiner"        # "planner", "refiner", "joint"
    anchor_source = "planner"     # "gt" (teacher forcing) or "planner" (for refiner training)
    pretrained_path = "/workspace/all_together/experiments/20260107_132607_planner/checkpoints/model_epoch_1700.pth" # Pretrained checkpoint path

    lr = 2e-4
    min_lr = 0.0
    weight_decay = 0.01
    epochs = 10000
    grad_clip_norm = 5.0
    
    # Loss Weights
    planner_loss_weight = 1.0
    diffusion_loss_weight = 1.0

    # --------------------------------------------------------------------------
    # 4. Logging & Visualization
    # --------------------------------------------------------------------------
    log_interval = 1
    vis_interval = 100
    ckpt_interval = 100
    max_vis = 1

    def to_dict(self):
        return {
            k: getattr(self, k)
            for k, v in Config.__dict__.items()
            if not k.startswith("__") and not callable(v)
        }


if __name__ == "__main__":
    config = Config()
    print(config.to_dict())