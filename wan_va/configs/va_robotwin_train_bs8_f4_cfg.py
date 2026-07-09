# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_robotwin_cfg import va_robotwin_cfg
import os

va_robotwin_train_bs8_f4_cfg = EasyDict(__name__='Config: VA robotwin train bs8 f4')
va_robotwin_train_bs8_f4_cfg.update(va_robotwin_cfg)

va_robotwin_train_bs8_f4_cfg.robotwin_data_root = '/mnt/data/share/data/robbyant/robotwin-clean-and-aug-lerobot/robotwin-clean-and-aug-lerobot'
va_robotwin_train_bs8_f4_cfg.dataset_path = os.path.join(va_robotwin_train_bs8_f4_cfg.robotwin_data_root, 'lerobot_robotwin_eef_clean_50')
va_robotwin_train_bs8_f4_cfg.empty_emb_path = os.path.join(va_robotwin_train_bs8_f4_cfg.robotwin_data_root, 'empty_emb.pt')
va_robotwin_train_bs8_f4_cfg.save_root = "/mnt/data/users/xianyi/EmbodyAi/lingbot-va/ckpts/train_out_robotwin_clean_bs8_f4"
va_robotwin_train_bs8_f4_cfg.enable_wandb = False
va_robotwin_train_bs8_f4_cfg.dataset_init_worker = 1
va_robotwin_train_bs8_f4_cfg.load_worker = 0
va_robotwin_train_bs8_f4_cfg.save_interval = 1000
va_robotwin_train_bs8_f4_cfg.gc_interval = 50
va_robotwin_train_bs8_f4_cfg.cfg_prob = 0.1

# Training parameters
va_robotwin_train_bs8_f4_cfg.learning_rate = 1e-5
va_robotwin_train_bs8_f4_cfg.beta1 = 0.9
va_robotwin_train_bs8_f4_cfg.beta2 = 0.95
va_robotwin_train_bs8_f4_cfg.weight_decay = 0.1
va_robotwin_train_bs8_f4_cfg.warmup_steps = 10
va_robotwin_train_bs8_f4_cfg.batch_size = 8
va_robotwin_train_bs8_f4_cfg.train_frame_num = 4
va_robotwin_train_bs8_f4_cfg.gradient_accumulation_steps = 1
va_robotwin_train_bs8_f4_cfg.num_steps = 5000
