# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .va_robotwin_train_cfg import va_robotwin_train_cfg


va_robotwin_contrastive_align_cfg = EasyDict(__name__="Config: VA robotwin contrastive align")
va_robotwin_contrastive_align_cfg.update(va_robotwin_train_cfg)

va_robotwin_contrastive_align_cfg.save_root = "./ckpts/train_out_contrastive_align_clean"
va_robotwin_contrastive_align_cfg.dataset_init_worker = 1
va_robotwin_contrastive_align_cfg.load_worker = 0

# Contrastive video-action transition alignment.
va_robotwin_contrastive_align_cfg.lambda_align = 0.1
va_robotwin_contrastive_align_cfg.align_action_frame_offset = 0
va_robotwin_contrastive_align_cfg.align_subspace_dim = 256
va_robotwin_contrastive_align_cfg.align_adapter_hidden_dim = 1024
va_robotwin_contrastive_align_cfg.align_adapter_depth = 2
va_robotwin_contrastive_align_cfg.align_adapter_dropout = 0.0
va_robotwin_contrastive_align_cfg.align_logit_scale_init = 1.0 / 0.07
va_robotwin_contrastive_align_cfg.align_learning_rate = va_robotwin_contrastive_align_cfg.learning_rate
va_robotwin_contrastive_align_cfg.align_weight_decay = va_robotwin_contrastive_align_cfg.weight_decay

# Inverse regularizer: video residual subspace -> action.
va_robotwin_contrastive_align_cfg.lambda_action_recon = 0.1
