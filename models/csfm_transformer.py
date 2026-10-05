import torch
import torch.nn as nn
from .vit_wrapper import OfficialViTEncoder
from .dit_wrapper import DiTContinuous
from .dit_crossattn_wrapper import DiTCrossAttnContinuous


class CSFMTransformerModel(nn.Module):
    """
    Unified transformer model supporting both adaLN and Cross-Attention DiT regressor backbones.
    Uses consistent naming (enc_dim, enc_depth, reg_dim, reg_depth, patch_size).
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        img_size = config.data.image_size
        channels = config.data.num_channels

        # Consistent Latent Dimension
        widen_factor = getattr(config.model, 'widen_factor', 4.0)
        latent_dim = getattr(config.model, 'latent_dim', int(64 * widen_factor))

        # Consistent Encoder Parameters
        patch_size = getattr(config.model, 'patch_size', getattr(config.model, 'vit_patch_size', 4)) 
        enc_dim = getattr(config.model, 'enc_dim', getattr(config.model, 'vit_embed_dim', 384))
        enc_depth = getattr(config.model, 'enc_depth', getattr(config.model, 'vit_depth', 12))
        enc_heads = getattr(config.model, 'enc_heads', getattr(config.model, 'vit_num_heads', 6))
        mlp_ratio = getattr(config.model, 'mlp_ratio', getattr(config.model, 'vit_mlp_ratio', 4.0)) 

        self.encoder = OfficialViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=channels,
            embed_dim=enc_dim,
            depth=enc_depth,
            num_heads=enc_heads,
            mlp_ratio=mlp_ratio,
            latent_dim=latent_dim,
            global_pool=getattr(config.model, 'vit_pool', 'token') 
        )

        # Consistent Regressor Parameters
        regressor_type = getattr(config.model, 'regressor_type', 'dit').lower()
        reg_dim = getattr(config.model, 'reg_dim', getattr(config.model, 'dit_hidden_size', 480))
        reg_depth = getattr(config.model, 'reg_depth', getattr(config.model, 'dit_depth', 14))
        reg_heads = getattr(config.model, 'reg_heads', getattr(config.model, 'dit_num_heads', 8))
        reg_patch_size = getattr(config.model, 'reg_patch_size', getattr(config.model, 'dit_patch_size', patch_size)) 

        if regressor_type in ['dit_crossattn', 'cross_attention']:
            self.regressor = DiTCrossAttnContinuous(
                input_size=img_size,
                patch_size=reg_patch_size,
                in_channels=channels,
                reg_dim=reg_dim,
                reg_depth=reg_depth,
                reg_heads=reg_heads,
                mlp_ratio=mlp_ratio,
                latent_dim=latent_dim
            )
        else:
            self.regressor = DiTContinuous(
                input_size=img_size,
                patch_size=reg_patch_size,
                in_channels=channels,
                hidden_size=reg_dim,
                depth=reg_depth,
                num_heads=reg_heads,
                mlp_ratio=mlp_ratio,
                latent_dim=latent_dim
            )

    def forward(self, x=None, time_cond=None, x0=None, z_cond=None, mode='flow'): 
        # Mode 1: Encoder representation extraction
        if mode in ['flow_enc', 'lejea']: 
            img = x0 if x0 is not None else x 
            enc_tuple = self.encoder(img) 
            return {'latent': enc_tuple[2], 'output': None} 

        # Mode 2: Continuous Flow / Regressor integration
        if mode in ['flow', 'cdae', 'ae']: 
            if z_cond is None: 
                if x0 is not None: 
                    z_cond = self.encoder(x0)[2] 
                else:
                    z_cond = self.encoder(x)[2] 
            if time_cond is None: 
                time_cond = torch.zeros(x.shape[0], device=x.device) 
            v_pred = self.regressor(x=x, t=time_cond, z=z_cond) 
            return {'output': v_pred, 'latent': z_cond} 

        raise NotImplementedError(f"Unsupported forward mode: '{mode}'") 