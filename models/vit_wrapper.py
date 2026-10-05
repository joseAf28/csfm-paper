import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import VisionTransformer



class OfficialViTEncoder(nn.Module):
    """
    Vision Transformer Encoder wrapping the official ViT implementation.
    Returns (None, rep, z) to maintain compatibility with eval_metrics.py:
        z = model.encoder(images, t=None)
    """
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_chans=3,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        latent_dim=128,
        global_pool='token'
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.global_pool = global_pool

        self.vit = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            global_pool=global_pool,
            num_classes=0
            )
        feat_dim = embed_dim

        self.projector = nn.Linear(feat_dim, latent_dim)

    def forward(self, x, t=None):

        feats = self.vit.forward_features(x)
        if self.global_pool == 'token' and feats.ndim == 3:
            rep = feats[:, 0]
        elif feats.ndim == 3:
            rep = feats.mean(dim=1)
        else:
            rep = feats

        z = self.projector(rep)
        return None, rep, z