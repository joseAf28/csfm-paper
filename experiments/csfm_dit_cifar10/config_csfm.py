import ml_collections
from expFinal.base_cifar10_large import get_base_config


def get_config():
    config = get_base_config()

    # -------------------------------------------------------------
    # Experiment Identity & Objective
    # -------------------------------------------------------------
    config.training.experiment_name = 'CSFM_ViT_DiT'
    config.training.loss_type = 'csfm'
    config.training.ssd_eta = 0.0

    # CSFM Physics & Decay Schedule (Exact original parameters)
    config.training.operators = ['illumination', 'color', 'freq']
    config.training.r_low = 6
    config.training.k_base = 3.25
    config.training.k_beta = 0.9
    config.training.decay_mode = 'hard_truncated_exponential'
    config.training.t1 = 5e-1

    # Latent Regularization (SIGReg)
    config.training.reg_mode = 'sigreg'
    config.training.lambda_reg = 1e-3

    # Time Sampling & Temporal Routing
    config.training.time_sampling_mode = 'uniform'
    config.training.loss_weighting_mode = 'none'
    config.training.t0 = 5e-2

    # -------------------------------------------------------------
    # Activate Transformer Backbone (ViT + DiT)
    # -------------------------------------------------------------
    config.model.architecture = 'transformer'
    config.model.arch_type = 'transformer'
    config.model.encoder_type = 'vit'
    config.model.regressor_type = 'dit'
    config.model.name = 'csfm_transformer'
    config.model.widen_factor = 4.0            # Latent dim: 64 * 4.0 = 256
    config.model.latent_dim = 256
    config.model.ema_rate = 0.999

    # Standardized ViT Encoder Parameters (~5.6M parameters)
    config.model.patch_size = 4
    config.model.enc_dim = 256
    config.model.enc_depth = 7
    config.model.enc_heads = 4
    config.model.mlp_ratio = 4.0
    config.model.pool_type = 'token'

    # Standardized DiT Regressor Parameters (~57.2M parameters)
    config.model.reg_patch_size = 2
    config.model.reg_dim = 512
    config.model.reg_depth = 12
    config.model.reg_heads = 8

    # -------------------------------------------------------------------------
    # Optimization Configuration for Transformers (ViT + DiT)
    # -------------------------------------------------------------------------
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = 'AdamW'
    optim.lr = 2e-4
    optim.encoder_lr_mult = 1.0
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.weight_decay = 1e-4
    optim.warmup = 5000
    optim.grad_clip = 1.0

    # Augmentation
    config.data.aug = 'none'

    return config