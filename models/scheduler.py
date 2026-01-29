import torch
import torch.nn.functional as F
import numpy as np

class NoiseScheduler:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device="cpu"):
        self.num_timesteps = num_timesteps
        self.device = device
        
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def add_noise(self, original_samples, noise, timesteps):
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].flatten()
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].flatten()

        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def step(self, model_output, timestep, sample):
        t = timestep
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_cumprod_t = self.alphas_cumprod[t]

        coef_eps = (1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)
        prev_sample = (1 / torch.sqrt(alpha_t)) * (sample - coef_eps * model_output)

        if t > 0:
            noise = torch.randn_like(sample)
            sigma_t = torch.sqrt(beta_t)
            prev_sample = prev_sample + sigma_t * noise
            
        return prev_sample

    def step_x0(self, pred_x0, timestep, sample):
        t = timestep
        
        alpha_cumprod_t = self.alphas_cumprod[t]
        alpha_cumprod_prev_t = self.alphas_cumprod_prev[t]
        beta_t = self.betas[t]
        
        pred_x0 = pred_x0.clamp(-1, 1)
        
        posterior_mean_coef1 = beta_t * torch.sqrt(alpha_cumprod_prev_t) / (1.0 - alpha_cumprod_t)
        posterior_mean_coef2 = (1.0 - alpha_cumprod_prev_t) * torch.sqrt(self.alphas[t]) / (1.0 - alpha_cumprod_t)
        
        posterior_mean = posterior_mean_coef1 * pred_x0 + posterior_mean_coef2 * sample
        
        if t > 0:
            noise = torch.randn_like(sample)
            posterior_variance = self.posterior_variance[t]
            posterior_log_variance_clipped = torch.log(posterior_variance.clamp(min=1e-20))
            
            prev_sample = posterior_mean + torch.exp(0.5 * posterior_log_variance_clipped) * noise
        else:
            prev_sample = posterior_mean
            
        return prev_sample
