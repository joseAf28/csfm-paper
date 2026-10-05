import ml_collections
from expFinal.base_cifar100_large import get_base_config


def get_config():
    config = get_base_config()

    # -------------------------------------------------------------
    # Experiment Identity & Objective
    # -------------------------------------------------------------
    config.training.experiment_name = 'OT_cifar100_ViT_DiT'
    config.training.loss_type = 'ot'
    config.training.ssd_eta = 0.0

    # Latent Regularization (SIGReg)
    config.training.reg_mode = 'sigreg'
    config.training.lambda_reg = 1e-3

    # Time Sampling & Temporal Routing
    config.training.time_sampling_mode = 'uniform'
    config.training.loss_weighting_mode = 'none'
    config.training.t0 = 5e-2


    config.model.architecture = 'cnn'
    config.model.arch_type = 'cnn'
    config.model.encoder_type = 'wrn'
    config.model.regressor_type = 'unet'
    config.model.name = 'ncsnpp'
    
    
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