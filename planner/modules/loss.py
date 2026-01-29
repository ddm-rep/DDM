
import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(
        self,
        planner_pred: torch.Tensor | None,
        future_latent: torch.Tensor | None,
        eps_pred: torch.Tensor | None,
        noise: torch.Tensor | None,
        attn_mask: torch.Tensor,
    ):
        if planner_pred is not None:
            target_ndim = planner_pred.ndim
        elif eps_pred is not None:
            target_ndim = eps_pred.ndim
        else:
            target_ndim = 4 # default

        if attn_mask.ndim == 1:
            mask = attn_mask.view(-1, 1, 1, 1)
        elif attn_mask.ndim == 2:
            if target_ndim == 5:
                mask = attn_mask.view(attn_mask.shape[0], attn_mask.shape[1], 1, 1, 1)
            else:
                mask = attn_mask.view(attn_mask.shape[0], attn_mask.shape[1], 1, 1)
        else:
            mask = attn_mask

        planner_loss = torch.tensor(0.0, device=attn_mask.device)
        if planner_pred is not None and future_latent is not None:
            diff = (planner_pred - future_latent) * mask
            numerator = diff.pow(2).sum()
            
            elements_per_sample = 1
            for dim in planner_pred.shape[mask.ndim:]:
                elements_per_sample *= dim
                
            denominator = mask.sum() * elements_per_sample + 1e-8
            planner_loss = numerator / denominator

        diffusion_loss = torch.tensor(0.0, device=attn_mask.device)
        if eps_pred is not None and noise is not None:
            diff = (eps_pred - noise) * mask
            
            elements_per_sample = 1
            for dim in eps_pred.shape[mask.ndim:]:
                elements_per_sample *= dim
                
            denominator = mask.sum() * elements_per_sample + 1e-8
            diffusion_loss = diff.pow(2).sum() / denominator

        total = (
            float(getattr(self.config, "planner_loss_weight", 1.0)) * planner_loss
            + float(getattr(self.config, "diffusion_loss_weight", 1.0)) * diffusion_loss
        )

        loss_dict = {
            "total": total,
            "planner": planner_loss,
            "diffusion": diffusion_loss,
        }
        return total, loss_dict
