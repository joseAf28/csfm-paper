# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# References:
# DiT:  https://github.com/facebookresearch/DiT/blob/main/models.py
# MAE:  https://github.com/facebookresearch/mae/blob/main/models_mae.py

import math
import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed

# Import core modular primitives from official models_dit.py
from .models_dit import (
    modulate,
    TimestepEmbedder,
    FinalLayer,
    get_2d_sincos_pos_embed,
)


#################################################################################
#                         Multi-Head Cross-Attention Layer                      #
#################################################################################

class MultiHeadCrossAttention(nn.Module):
    """
    Standard Multi-Head Cross-Attention module querying the latent representation z.
    Adheres strictly to the scaled dot-product attention formulation and timm conventions.
    """
    def __init__(self, dim, context_dim, num_heads=8, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}."

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim, bias=qkv_bias)

    def forward(self, x, context):
        """
        Args:
          x: Spatial patch sequence (B, N, D)
          context: Encoder representation sequence (B, M, D_c) or pooled vector (B, 1, D_c)
        Returns:
          Cross-attended token representation of shape (B, N, D)
        """
        B, N, D = x.shape
        M = context.shape[1]

        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(context).reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(context).reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, D)
        out = self.out_proj(out)
        return out


#################################################################################
#                   Dual adaLN + Cross-Attention DiT Block                      #
#################################################################################

class CrossAttentionDiTBlock(nn.Module):
    """
    DiT block that integrates both conditioning mechanisms:
    1. adaLN-Zero for global continuous time (t) modulation across Self-Attention & MLP.
    2. Multi-Head Cross-Attention querying semantic representation (z) with zero-initialized gating.
    """
    def __init__(self, hidden_size, num_heads, context_dim=256, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        # 1. Self-Attention (Spatial interactions)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)

        # 2. Cross-Attention (Semantic query over z)
        self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.cross_attn = MultiHeadCrossAttention(
            dim=hidden_size,
            context_dim=context_dim,
            num_heads=num_heads,
            qkv_bias=True
        )
        # Learnable gate initialized to 0 for exact identity initialization at step 0
        self.gate_cross = nn.Parameter(torch.zeros(hidden_size))

        # 3. Feedforward Pointwise MLP
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        # 4. adaLN modulation driven by timestep t (6 scale/shift/gate parameters)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, t_emb, z_context):
        """
        x: (B, N, hidden_size)
        t_emb: (B, hidden_size) -> Output from TimestepEmbedder(t)
        z_context: (B, M, context_dim) -> Spatial tokens or pooled representation from encoder
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t_emb).chunk(6, dim=1)

        # Sublayer 1: Self-Attention with temporal adaLN-Zero
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))

        # Sublayer 2: Multi-Head Cross-Attention querying z
        ca_out = self.cross_attn(self.norm_cross(x), z_context)
        x = x + self.gate_cross.unsqueeze(0).unsqueeze(1) * ca_out

        # Sublayer 3: Feedforward MLP with temporal adaLN-Zero
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x


#################################################################################
#                           Full DiTCrossAttention Model                        #
#################################################################################

class DiTCrossAttention(nn.Module):
    """
    Diffusion / Flow Vector Field Regressor matching the CSFMModel interface.
    Employs adaLN for continuous time t and Multi-Head Cross-Attention for latent z.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=4,
        in_channels=3,
        hidden_size=480,
        depth=14,
        num_heads=8,
        mlp_ratio=4.0,
        latent_dim=256,
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        # Patch embedding and temporal embedder (imported from models_dit.py)
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        num_patches = self.x_embedder.num_patches
        # Fixed 2D sin-cos position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # Transformer blocks with Cross-Attention + adaLN
        self.blocks = nn.ModuleList([
            CrossAttentionDiTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                context_dim=latent_dim,
                mlp_ratio=mlp_ratio
            )
            for _ in range(depth)
        ])

        # Final projection layer (imported from models_dit.py)
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # 1. Basic linear initialization matching official DiT
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # 2. Initialize (and freeze) pos_embed with 2D sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # 3. Initialize patch_embed like nn.Linear
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # 4. Initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # 5. Zero-out adaLN modulation layers across blocks
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            # Zero-out the cross-attention gate for exact identity initialization
            nn.init.constant_(block.gate_cross, 0)

        # 6. Zero-out output final layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, time_cond, z_cond, mode='flow'):
        """
        Unified forward pass adhering strictly to CSFMModel requirements.
        x: (N, C, H, W) continuous velocity / sample state
        time_cond: (N,) continuous timesteps in [0, 1]
        z_cond: (N, D_c) or (N, M, D_c) encoder latent representation
        """
        if time_cond.ndim > 1:
            time_cond = time_cond.flatten()

        # Tokenize and inject spatial position embeddings
        h = self.x_embedder(x) + self.pos_embed

        # Fourier timestep embedding
        t_emb = self.t_embedder(time_cond)

        # Prepare latent context sequence: (B, M, D_c)
        if z_cond.ndim == 2:
            z_context = z_cond.unsqueeze(1)
        else:
            z_context = z_cond

        # Process through Cross-Attention DiT blocks
        for block in self.blocks:
            h = block(h, t_emb, z_context)

        # Final adaLN unprojection
        h = self.final_layer(h, t_emb)
        v_pred = self.unpatchify(h)

        return v_pred