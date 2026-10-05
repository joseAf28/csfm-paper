# models/dit_crossattn_wrapper.py
import torch
import torch.nn as nn
from .dit_crossattn import DiTCrossAttention


class DiTCrossAttnContinuous(nn.Module):
    """
    Continuous Vector Field Regressor using Multi-Head Cross-Attention.
    Adopts consistent parameter naming (reg_dim, reg_depth, reg_heads, patch_size).
    """
    def __init__(
        self,
        input_size=32,
        patch_size=4,
        in_channels=3,
        reg_dim=480,
        reg_depth=14,
        reg_heads=8,
        mlp_ratio=4.0,
        latent_dim=256
    ):
        super().__init__()
        self.regressor = DiTCrossAttention(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=reg_dim,
            depth=reg_depth,
            num_heads=reg_heads,
            mlp_ratio=mlp_ratio,
            latent_dim=latent_dim
        )

    def forward(self, x, t, z):
        """
        Maintains identical argument signatures:
          x: [B, C, H, W]
          t: [B]
          z: [B, latent_dim]
        """
        return self.regressor(x=x, time_cond=t, z_cond=z)