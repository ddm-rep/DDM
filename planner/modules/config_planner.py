import os


class ConfigPlanner:
    data_root = "/workspace/data/processed_32"
    refiner_ckpt_path = None
    
    experiments_root = "/workspace/planner/experiments"

    batch_size = 32
    num_workers = 8
    
    max_seq_len = 100
    target_res = 32
    cond_frames = 5
    
    latent_dim = 3072
    
    planner_dim = 1024
    planner_heads = 16
    planner_layers = 12
    planner_dropout = 0.1

    lr = 2e-4
    min_lr = 1e-6
    weight_decay = 0.01
    epochs = 1000
    grad_clip_norm = 5.0
    
    planner_loss_weight = 1.0

    ckpt_interval = 50
    vis_interval = 50
    vis_frames = 100
    vis_fps = 10

    def to_dict(self):
        return {
            k: getattr(self, k)
            for k, v in ConfigPlanner.__dict__.items()
            if not k.startswith("__") and not callable(v)
        }


if __name__ == "__main__":
    config = ConfigPlanner()
    print(config.to_dict())
