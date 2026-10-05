import torch
import torch.nn as nn
from .models_dit import (
    PatchEmbed,
    TimestepEmbedder,
    DiTBlock,
    FinalLayer,
    get_2d_sincos_pos_embed
)

class LatentEmbedder(nn.Module):
    """
    Embeds continuous latent vectors z into the hidden dimension c.
    Replaces discrete class-label embedding for representation conditioning.
    """
    def __init__(self, latent_dim, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True)
        )

    def forward(self, z):
        return self.mlp(z)


class DiTContinuous(nn.Module):
    """
    Continuous Vector Field Regressor using the official Meta DiT architecture.
    Conditioned on diffusion timestep t and continuous latent vector z.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=3,
        hidden_size=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        latent_dim=128
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.z_embedder = LatentEmbedder(latent_dim, hidden_size)

        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        grid_size = int(self.x_embedder.num_patches ** 0.5)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.z_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.z_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward(self, x, t, z):
        # x: [B, C, H, W], t: [B], z: [B, latent_dim]
        x = self.x_embedder(x) + self.pos_embed
        c = self.t_embedder(t) + self.z_embedder(z)

        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        v_pred = self.unpatchify(x)
        return v_pred