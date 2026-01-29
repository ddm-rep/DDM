import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)

class LatentPlanner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.latent_dim = config.latent_dim
        self.planner_dim = config.planner_dim
        self.planner_heads = config.planner_heads
        self.planner_layers = config.planner_layers
        self.planner_dropout = config.planner_dropout
        self.max_len = config.max_seq_len

        self.in_proj = nn.Linear(self.latent_dim, self.planner_dim)
        self.out_proj = nn.Linear(self.planner_dim, self.latent_dim)

        self.bos = nn.Parameter(torch.zeros(1, self.planner_dim))
        nn.init.normal_(self.bos, std=0.02)

        self.pe = PositionalEncoding(self.planner_dim, max_len=self.max_len + 16)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.planner_dim,
            nhead=self.planner_heads,
            dim_feedforward=self.planner_dim * 4,
            dropout=self.planner_dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.planner_layers)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones((T, T), device=device, dtype=torch.bool), diagonal=1)

    def forward(self, seq: torch.Tensor, context: torch.Tensor = None, attn_mask: torch.Tensor = None):
        device = seq.device
        if seq.ndim != 5:
             raise ValueError(f"Expected seq to be [B, T, C, H, W], got {seq.shape}")
        
        B, T, C, H, W = seq.shape
        seq_flat = seq.reshape(B, T, -1)
        
        x = self.in_proj(seq_flat)
        
        bos = self.bos.expand(B, 1, -1)
        decoder_input = torch.cat([bos, x], dim=1)
        
        decoder_input = self.pe(decoder_input)
        
        causal_mask = self._causal_mask(T + 1, device)
        
        key_padding = None
        if attn_mask is not None:
            bos_mask = torch.ones(B, 1, device=device, dtype=attn_mask.dtype)
            full_mask = torch.cat([bos_mask, attn_mask], dim=1)
            key_padding = (full_mask == 0)
            
        if context is None:
             context = torch.zeros(B, 1, self.planner_dim, device=device)
             
        out = self.decoder(
            tgt=decoder_input,
            memory=context,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=key_padding
        )
        
        pred_flat = self.out_proj(out)
        pred = pred_flat.view(B, T + 1, C, H, W)
        
        return pred

    @torch.no_grad()
    def sample(self, current_frame: torch.Tensor, num_frames: int, device: torch.device, context: torch.Tensor = None) -> torch.Tensor:
         B, C, H, W = current_frame.shape
         
         bos = self.bos.expand(B, 1, -1)
         f0_flat = self.in_proj(current_frame.view(B, 1, -1))
         decoder_input = torch.cat([bos, f0_flat], dim=1)
         
         preds = [current_frame]
         
         for t in range(num_frames - 1):
             x = self.pe(decoder_input)
             causal_mask = self._causal_mask(decoder_input.size(1), device)
             
             if context is None:
                 ctx = torch.zeros(B, 1, self.planner_dim, device=device)
             else:
                 ctx = context

             out = self.decoder(tgt=x, memory=ctx, tgt_mask=causal_mask)
             
             last_out = out[:, -1, :]
             next_flat = self.out_proj(last_out)
             next_frame = next_flat.view(B, C, H, W)
             
             preds.append(next_frame)
             
             next_in = self.in_proj(next_flat).unsqueeze(1)
             decoder_input = torch.cat([decoder_input, next_in], dim=1)
             
         return torch.stack(preds, dim=1)

if __name__ == "__main__":
    from config import Config
    config = Config()

    B, C, H, W = 2, 4, 16, 16
    T = 10
    
    config.latent_dim = C*H*W
    config.planner_dim = 128
    
    planner = LatentPlanner(config)

    # Sequence Input: [B, T, C, H, W]
    x = torch.randn(B, T, C, H, W)
    
    # Forward Test
    pred = planner(x) 
    print(f"Seq Input: {x.shape}")
    print(f"Seq Pred: {pred.shape}") # Should be [B, T+1, C, H, W]
    
    # Sample Test
    f0 = x[:, 0]
    sample = planner.sample(f0, num_frames=5, device=x.device)
    print(f"Sampled sequence: {sample.shape}") # Should be [B, 5, C, H, W]