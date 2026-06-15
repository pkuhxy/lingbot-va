# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .va_libero_train_cfg import va_libero_train_cfg


va_libero_latent_posttrain_cfg = EasyDict(__name__="Config: WAM latent posttrain libero")
va_libero_latent_posttrain_cfg.update(va_libero_train_cfg)

va_libero_latent_posttrain_cfg.enable_wandb = False
va_libero_latent_posttrain_cfg.save_root = "./train_out_wam_latent_libero"
va_libero_latent_posttrain_cfg.dataset_init_worker = 8
va_libero_latent_posttrain_cfg.load_worker = 8
va_libero_latent_posttrain_cfg.save_interval = 500

va_libero_latent_posttrain_cfg.latent_hidden_source = "embed"
va_libero_latent_posttrain_cfg.latent_hidden_attn_mode = "torch"
va_libero_latent_posttrain_cfg.video_delta_mode = "last_first"
va_libero_latent_posttrain_cfg.stage1_chunk_size = va_libero_latent_posttrain_cfg.frame_chunk_size
va_libero_latent_posttrain_cfg.stage1_window_size = va_libero_latent_posttrain_cfg.attn_window

va_libero_latent_posttrain_cfg.subspace_dim = 256
va_libero_latent_posttrain_cfg.adapter_hidden_dim = 1024
va_libero_latent_posttrain_cfg.adapter_depth = 2
va_libero_latent_posttrain_cfg.adapter_dropout = 0.0

va_libero_latent_posttrain_cfg.latent_batch_size = 8
va_libero_latent_posttrain_cfg.gradient_accumulation_steps = 1
va_libero_latent_posttrain_cfg.num_steps = 5000
va_libero_latent_posttrain_cfg.latent_learning_rate = 1e-4
va_libero_latent_posttrain_cfg.latent_weight_decay = 1e-4
va_libero_latent_posttrain_cfg.max_grad_norm = 2.0
va_libero_latent_posttrain_cfg.warmup_steps = 100

va_libero_latent_posttrain_cfg.lambda_recon_action = 1.0
va_libero_latent_posttrain_cfg.lambda_inverse = 1.0
va_libero_latent_posttrain_cfg.lambda_counterfactual = 0.2
va_libero_latent_posttrain_cfg.lambda_relational = 0.0
va_libero_latent_posttrain_cfg.counterfactual_tau = 0.1
va_libero_latent_posttrain_cfg.counterfactual_margin = None
va_libero_latent_posttrain_cfg.counterfactual_queue_size = 2048
va_libero_latent_posttrain_cfg.huber_beta = 0.1
