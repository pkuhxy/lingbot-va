# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import torch
from easydict import EasyDict

va_shared_cfg = EasyDict()

va_shared_cfg.host = '0.0.0.0'
va_shared_cfg.port = 29536

va_shared_cfg.param_dtype = torch.bfloat16
va_shared_cfg.save_root = './train_out'

va_shared_cfg.patch_size = (1, 2, 2)

va_shared_cfg.enable_offload = False

# WAM internal World2Act posttraining. Default keeps the original SFT behavior.
va_shared_cfg.wam_posttrain_stage = 'sft'
va_shared_cfg.wam_adapter_path = None
va_shared_cfg.wam_latent_dim = 256
va_shared_cfg.wam_adapter_hidden_dim = None
va_shared_cfg.wam_adapter_dropout = 0.0
va_shared_cfg.wam_video_repr_key = 'latent_clean'
va_shared_cfg.wam_action_repr_key = 'action_clean'
va_shared_cfg.wam_video_delta_mode = 'last_first'
va_shared_cfg.wam_latent_noisy_cond_prob = 0.0
va_shared_cfg.wam_recon_action_weight = 1.0
va_shared_cfg.wam_inverse_weight = 1.0
va_shared_cfg.wam_counterfactual_weight = 0.1
va_shared_cfg.wam_counterfactual_tau = 0.1
va_shared_cfg.wam_relational_weight = 0.0
va_shared_cfg.wam_cf_queue_size = 1024
va_shared_cfg.wam_cf_gather_distributed = True
va_shared_cfg.wam_latent_consistency_weight = 1.0
va_shared_cfg.wam_stage2_action_loss_weight = 1.0
va_shared_cfg.wam_stage2_video_loss_weight = 0.0
va_shared_cfg.wam_stage2_counterfactual_weight = 0.0
va_shared_cfg.wam_stage2_trainable_keywords = (
    'action_embedder',
    'condition_embedder_action',
    'action_proj_out',
)
va_shared_cfg.wam_save_frozen_transformer = False
