import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if time_emb_dim is not None
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = time_emb.view(*time_emb.shape, 1, 1)
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

class Downsample(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = dim_out or dim
        self.conv = nn.Conv2d(dim, dim_out, 3, stride=2, padding=1)
        
    def forward(self, x):
        return self.conv(x)

class UpsampleConv(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = dim_out or dim
        self.conv = nn.Conv2d(dim, dim_out, 3, padding=1)

    def forward(self, x, target_size=None):
        if target_size is None:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        else:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return self.conv(x)

class AsymmetricUNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        model_channels=128,
        channel_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        time_emb_dim=512,
    ):
        super().__init__()
        self.model_channels = model_channels
        self.num_levels = len(channel_mult)
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        self.downs = nn.ModuleList([])
        ch = model_channels
        curr_mult = 1
        
        self.enc_channels = [ch] 

        for i, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                self.downs.append(ResnetBlock(ch, out_ch, time_emb_dim=time_emb_dim))
                ch = out_ch
                self.enc_channels.append(ch)
            
            if i != len(channel_mult) - 1:
                self.downs.append(Downsample(ch))
                self.enc_channels.append(ch)

        self.mid_block1 = ResnetBlock(ch, ch, time_emb_dim=time_emb_dim)
        self.mid_block2 = ResnetBlock(ch, ch, time_emb_dim=time_emb_dim)

        self.ups = nn.ModuleList([])
        
        rev_channel_mult = list(reversed(channel_mult))
        
        for i, mult in enumerate(rev_channel_mult):
            out_ch = model_channels * mult
            
            for _ in range(num_res_blocks + 1):
                skip_ch = self.enc_channels.pop()
                self.ups.append(ResnetBlock(ch + skip_ch, out_ch, time_emb_dim=time_emb_dim))
                ch = out_ch
            
            if i != len(channel_mult) - 1:
                self.ups.append(UpsampleConv(ch))

        self.final_norm = nn.GroupNorm(8, ch)
        self.final_conv = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(self, x, t, target_shape=None):
        """
        x: (B, C, h, w)
        target_shape: (H, W) - Final Output Resolution
        """
        t_emb = self.time_mlp(t)
        
        h_list = []
        
        h = self.input_conv(x)
        h_list.append(h)
        
        for module in self.downs:
            if isinstance(module, ResnetBlock):
                h = module(h, t_emb)
                h_list.append(h)
            elif isinstance(module, Downsample):
                h = module(h)
                h_list.append(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_block2(h, t_emb)

        if target_shape is None:
            target_H, target_W = x.shape[2] * (2**(self.num_levels-1)), x.shape[3] * (2**(self.num_levels-1))
        else:
            target_H, target_W = target_shape

        resolutions = []
        curr_h, curr_w = target_H, target_W
        for _ in range(self.num_levels):
            resolutions.append((curr_h, curr_w))
            curr_h, curr_w = curr_h // 2, curr_w // 2
        
        resolutions.reverse()
        
        res_idx = 0
        
        for module in self.ups:
            if isinstance(module, UpsampleConv):
                res_idx += 1
                target_res = resolutions[res_idx] if res_idx < len(resolutions) else (target_H, target_W)
                h = module(h, target_size=target_res)
            
            elif isinstance(module, ResnetBlock):
                skip = h_list.pop()
                
                if skip.shape[2:] != h.shape[2:]:
                    skip = F.interpolate(skip, size=h.shape[2:], mode="bilinear", align_corners=False)
                
                h = torch.cat([h, skip], dim=1)
                h = module(h, t_emb)

        if h.shape[2:] != (target_H, target_W):
             h = F.interpolate(h, size=(target_H, target_W), mode="bilinear", align_corners=False)

        h = self.final_norm(h)
        h = F.silu(h)
        h = self.final_conv(h)

        return h
