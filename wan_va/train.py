# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import argparse
import os
import sys
from pathlib import Path
import wandb

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from safetensors.torch import save_file, load_file
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS
from distributed.fsdp import shard_model, apply_ac
from distributed.util import (
    _configure_model, 
    init_distributed, 
    dist_mean, 
    dist_max
)
from einops import rearrange
from modules.utils import (
    load_transformer,
)
from utils import (
    init_logger, 
    logger, 
    get_mesh_id, 
    sample_timestep_id,
    data_seq_to_patch,
    warmup_constant_lambda,
    FlowMatchScheduler
)

from dataset import MultiLatentLeRobotDataset
import gc


class AdapterMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout=0.0, output_norm=True):
        super().__init__()
        layers = [
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))
        if output_norm:
            layers.append(nn.LayerNorm(output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.float())


class InternalWorld2ActAdapters(nn.Module):
    """Training-only probes for controllable video dynamics and action-causal latents."""

    def __init__(self, hidden_dim, action_dim, latent_dim=256, adapter_hidden_dim=None, dropout=0.0):
        super().__init__()
        adapter_hidden_dim = adapter_hidden_dim or max(latent_dim * 4, hidden_dim // 2)
        self.video_adapter = AdapterMLP(
            hidden_dim, latent_dim, adapter_hidden_dim, dropout=dropout, output_norm=True)
        self.action_adapter = AdapterMLP(
            hidden_dim, latent_dim, adapter_hidden_dim, dropout=dropout, output_norm=True)
        self.action_decoder = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, adapter_hidden_dim),
            nn.SiLU(),
            nn.Linear(adapter_hidden_dim, action_dim),
        )
        self.inverse_decoder = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, adapter_hidden_dim),
            nn.SiLU(),
            nn.Linear(adapter_hidden_dim, action_dim),
        )

    def encode_video(self, video_delta):
        return self.video_adapter(video_delta)

    def encode_action_tokens(self, action_hidden):
        return self.action_adapter(action_hidden)

    def decode_action_tokens(self, action_codes):
        return self.action_decoder(action_codes.float())

    def decode_inverse_action(self, video_code):
        return self.inverse_decoder(video_code.float())


class Trainer:
    def __init__(self, config):
        if config.enable_wandb and config.rank == 0:
            wandb.login(host=os.environ['WANDB_BASE_URL'], key=os.environ['WANDB_API_KEY'])
            self.wandb = wandb
            self.wandb.init(
                entity=os.environ["WANDB_TEAM_NAME"],
                project=os.getenv("WANDB_PROJECT", "va_robotwin"),
                # dir=log_dir,
                config=config,
                mode="online",
                name='test_lln'
                # name=os.path.basename(os.path.normpath(job_config.job.dump_folder))
            )
            logger.info("WandB logging enabled")
        self.step = 0
        self.config = config
        self.device = torch.device(f"cuda:{config.local_rank}")
        self.dtype = config.param_dtype
        self.patch_size = config.patch_size
        self.wam_stage = getattr(config, 'wam_posttrain_stage', 'sft')
        self.use_wam_latents = self.wam_stage in ('stage1_adapter', 'stage2_action_posttrain')
        self.wam_adapters = None
        self.wam_action_bank = None

        # Load models
        logger.info("Loading models...")

        # Load and shard transformer with FSDP
        logger.info("Loading transformer...")

        if hasattr(config, 'resume_from') and config.resume_from:
            transformer_path = os.path.join(config.resume_from, 'transformer')
            if config.rank == 0:
                logger.info(f"Resuming from checkpoint: {transformer_path}")
        else:
            transformer_path = os.path.join(config.wan22_pretrained_model_name_or_path, 'transformer')

        self.transformer = load_transformer(
            transformer_path,
            torch_dtype=torch.float32,
            torch_device='cpu',
            attn_mode="flex"
        )

        logger.info("Setting up activation checkpointing ...")
        apply_ac(self.transformer)

        logger.info("Setting up FSDP...")
        shard_fn = shard_model
        self.transformer = _configure_model(
            model=self.transformer,
            shard_fn=shard_fn,
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=False,
        )
        self._setup_posttraining_modules()

        trainable_params = [p for p in self.transformer.parameters() if p.requires_grad]
        if self.wam_adapters is not None:
            trainable_params.extend([p for p in self.wam_adapters.parameters() if p.requires_grad])
        if len(trainable_params) == 0:
            raise ValueError(f"No trainable parameters found for wam_posttrain_stage={self.wam_stage}")
        self.optimized_parameters = trainable_params

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.optimized_parameters,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=1e-8,
            weight_decay=config.weight_decay,
            fused=True,
            foreach=False,
        )

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, 
            lr_lambda=lambda step: warmup_constant_lambda(step, warmup_steps=config.warmup_steps))

        # Setup dataloaders
        logger.info("Setting up datasets...")
        train_dataset = MultiLatentLeRobotDataset(config=config)
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=config.world_size,
            rank=config.rank,
            shuffle=True,
            seed=42
        ) if config.world_size > 1 else None
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=(train_sampler is None), 
            num_workers=config.load_worker,
            sampler=train_sampler,
        )

        self.train_scheduler_latent = FlowMatchScheduler(shift=self.config.snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_latent.set_timesteps(1000, training=True)
        self.train_scheduler_action = FlowMatchScheduler(shift=self.config.action_snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_action.set_timesteps(1000, training=True)

        self.save_dir = Path(config.save_root) / "checkpoints"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
        self.train_loader_iter = None
        # if hasattr(config, 'resume_from') and config.resume_from:
        #     self._load_training_state(config.resume_from)

    def _model_config_value(self, key, default=None):
        model_config = self.transformer.config
        if isinstance(model_config, dict):
            return model_config.get(key, default)
        return getattr(model_config, key, default)

    def _adapter_module(self):
        if isinstance(self.wam_adapters, DistributedDataParallel):
            return self.wam_adapters.module
        return self.wam_adapters

    def _resolve_wam_adapter_path(self):
        adapter_path = getattr(self.config, 'wam_adapter_path', None)
        if adapter_path:
            return Path(adapter_path)

        resume_from = getattr(self.config, 'resume_from', None)
        if not resume_from:
            return None

        checkpoint_dir = Path(resume_from)
        for candidate in (
            checkpoint_dir / "adapters" / "wam_adapters.pt",
            checkpoint_dir / "wam_adapters.pt",
        ):
            if candidate.exists():
                return candidate
        return None

    def _load_wam_adapters(self, adapter_path):
        if adapter_path is None:
            return
        checkpoint = torch.load(adapter_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        self._adapter_module().load_state_dict(state_dict, strict=True)
        if self.config.rank == 0:
            logger.info(f"Loaded WAM adapters from {adapter_path}")

    def _setup_posttraining_modules(self):
        valid_stages = ('sft', 'stage1_adapter', 'stage2_action_posttrain')
        if self.wam_stage not in valid_stages:
            raise ValueError(f"Unsupported wam_posttrain_stage={self.wam_stage}; expected one of {valid_stages}")

        if self.wam_stage == 'sft':
            self.transformer.train()
            self.transformer.requires_grad_(True)
            return

        hidden_dim = int(
            self._model_config_value('num_attention_heads') *
            self._model_config_value('attention_head_dim')
        )
        action_dim = int(self._model_config_value('action_dim', getattr(self.config, 'action_dim', 30)))
        latent_dim = int(getattr(self.config, 'wam_latent_dim', 256))
        adapter_hidden_dim = getattr(self.config, 'wam_adapter_hidden_dim', None)
        dropout = float(getattr(self.config, 'wam_adapter_dropout', 0.0))
        self.wam_adapters = InternalWorld2ActAdapters(
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
            adapter_hidden_dim=adapter_hidden_dim,
            dropout=dropout,
        ).to(self.device)

        adapter_path = self._resolve_wam_adapter_path()
        if adapter_path is not None:
            self._load_wam_adapters(adapter_path)
        elif self.wam_stage == 'stage2_action_posttrain':
            raise ValueError(
                "stage2_action_posttrain requires --wam-adapter-path or a resume_from checkpoint "
                "containing adapters/wam_adapters.pt"
            )

        if self.wam_stage == 'stage1_adapter':
            self.transformer.eval()
            self.transformer.requires_grad_(False)
            self.wam_adapters.train()
            self.wam_adapters.requires_grad_(True)
            if self.config.world_size > 1:
                self.wam_adapters = DistributedDataParallel(
                    self.wam_adapters,
                    device_ids=[self.config.local_rank],
                    output_device=self.config.local_rank,
                )
        else:
            self.transformer.train()
            self.transformer.requires_grad_(False)
            trainable_keywords = getattr(
                self.config,
                'wam_stage2_trainable_keywords',
                ('action_embedder', 'condition_embedder_action', 'action_proj_out'),
            )
            for name, param in self.transformer.named_parameters():
                if any(keyword in name for keyword in trainable_keywords):
                    param.requires_grad_(True)

            self.wam_adapters.eval()
            self.wam_adapters.requires_grad_(False)

        if self.config.rank == 0:
            trainable_transformer = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
            trainable_adapters = 0 if self.wam_adapters is None else sum(
                p.numel() for p in self.wam_adapters.parameters() if p.requires_grad)
            logger.info(
                f"WAM posttraining stage={self.wam_stage}, "
                f"trainable transformer params={trainable_transformer:,}, "
                f"trainable adapter params={trainable_adapters:,}"
            )
    
    def _get_next_batch(self):
        """Get next batch from iterator, reset if epoch is finished."""
        if self.train_loader_iter is None:
            self.train_loader_iter = iter(self.train_loader)
        
        try:
            batch = next(self.train_loader_iter)
        except StopIteration:
            # Reset sampler and iterator when epoch finishes
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(self.train_loader.sampler.epoch + 1)
            self.train_loader_iter = iter(self.train_loader)
            batch = next(self.train_loader_iter)
        
        return batch

    @torch.no_grad()
    def _add_noise(self, latent, train_scheduler, action_mask=False, action_mode=False, noisy_cond_prob=0.):
        B, C, F, H, W = latent.shape

        timestep_ids = sample_timestep_id(batch_size=F, num_train_timesteps=train_scheduler.num_train_timesteps)
        noise = torch.zeros_like(latent).normal_()
        timesteps = train_scheduler.timesteps[timestep_ids].to(device=self.device)
        noisy_latents =train_scheduler.add_noise(latent, noise, timesteps, t_dim=2)
        targets =train_scheduler.training_target(latent, noise, timesteps)

        patch_f, patch_h, patch_w = self.patch_size
        if action_mode:
            patch_f = patch_h = patch_w = 1
        
        latent_grid_id = get_mesh_id(
            latent.shape[-3] // patch_f,  # F
            latent.shape[-2] // patch_h,  # H
            latent.shape[-1] // patch_w,  # W
            t=1 if action_mode else 0,  # 1 for action mode (0 for latent), not used
            f_w=1,
            f_shift=0,
            action=action_mode
        ).to(self.device)  # shape: [4, seq_len]
        latent_grid_id = latent_grid_id[None].repeat(B, 1, 1)

        if torch.rand(1).item() < noisy_cond_prob:
            cond_timestep_ids = sample_timestep_id(
                    batch_size=F,
                    min_timestep_bd=0.5, 
                    max_timestep_bd=1.0, 
                    num_train_timesteps=train_scheduler.num_train_timesteps,
                )
            noise = torch.zeros_like(latent).normal_()
            cond_timesteps = train_scheduler.timesteps[cond_timestep_ids].to(device=self.device)
            latent = train_scheduler.add_noise(latent, noise, cond_timesteps, t_dim=2)
        else:
            cond_timesteps = torch.zeros_like(timesteps)

        if action_mask is not None:
            noisy_latents *= action_mask.float()
            targets *= action_mask.float()
            latent *= action_mask.float()

        return dict(
            timesteps=timesteps[None].repeat(B, 1),
            noisy_latents=noisy_latents,
            targets=targets,
            latent=latent,
            cond_timesteps=cond_timesteps[None].repeat(B, 1),
            grid_id=latent_grid_id,
        )

    @torch.no_grad()
    def _prepare_input_dict(self, batch_dict):
        """Prepare input dict following infer code pattern from wan_va_server.py."""
        # Generate grid_id following infer code (no batch dimension yet)
        # For action mode: get_mesh_id(shape[-3], shape[-2], shape[-1], t=1, f_w=1, f_shift, action=True)
        latent_noisy_cond_prob = getattr(self.config, 'latent_noisy_cond_prob', 0.5)
        if self.use_wam_latents:
            latent_noisy_cond_prob = getattr(self.config, 'wam_latent_noisy_cond_prob', 0.0)

        latent_dict = self._add_noise(
            latent=batch_dict['latents'], 
            train_scheduler=self.train_scheduler_latent, 
            action_mask=None, 
            action_mode=False,
            noisy_cond_prob=latent_noisy_cond_prob)
        
        action_dict = self._add_noise(
            latent=batch_dict['actions'], 
            train_scheduler=self.train_scheduler_action, 
            action_mask=batch_dict['actions_mask'], 
            action_mode=True,
            noisy_cond_prob=0.0)

        latent_dict['text_emb'] = batch_dict['text_emb']
        action_dict['text_emb'] = batch_dict['text_emb']
        action_dict['actions_mask'] = batch_dict['actions_mask']

        input_dict = {
            'latent_dict': latent_dict,
            'action_dict': action_dict,
            'chunk_size': torch.randint(1, 5, (1,)).item(),
            'window_size': torch.randint(4, 65, (1,)).item(),
        }
        return input_dict

    def convert_input_format(self, input_dict):
        """Convert input dict to match transformer input format if needed."""
        for key, value in input_dict.items():
            input_dict[key] = value.to(self.device)#.to(self.dtype)
        return input_dict

    def compute_loss(self,
        input_dict,
        pred
    ):
        latent_pred, action_pred = pred[:2]
        action_pred = rearrange(action_pred, 'b (f n) c -> b c f n 1', f=input_dict['action_dict']['targets'].shape[-3])
        latent_pred = data_seq_to_patch(
                        self.patch_size, latent_pred,
                        input_dict['latent_dict']['targets'].shape[-3], input_dict['latent_dict']['targets'].shape[-2],
                        input_dict['latent_dict']['targets'].shape[-1], batch_size=latent_pred.shape[0])
        Bn, Fn = input_dict['latent_dict']['timesteps'].shape
        latent_loss_weight = self.train_scheduler_latent.training_weight(input_dict['latent_dict']['timesteps'].flatten()).reshape(Bn, Fn)
        action_loss_weight = self.train_scheduler_action.training_weight(input_dict['action_dict']['timesteps'].flatten()).reshape(Bn, Fn)

        # Frame-wise video loss calculation
        latent_loss = F.mse_loss(latent_pred.float(), input_dict['latent_dict']['targets'].float().detach(), reduction='none')
        latent_loss = latent_loss * latent_loss_weight[:, None, :, None, None]
        # Permute to (B, F, H, W, C) and flatten to (B*F, H*W*C)
        latent_loss = latent_loss.permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        latent_loss = latent_loss.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        # Sum per frame and compute mask per frame
        latent_loss_per_frame = latent_loss.sum(dim=1)  # (B*F,)
        latent_mask_per_frame = torch.ones_like(latent_loss).sum(dim=1)  # (B*F,)
        latent_loss = (latent_loss_per_frame / (latent_mask_per_frame + 1e-6)).mean()

        # Frame-wise action loss calculation
        action_loss = F.mse_loss(action_pred.float(), input_dict['action_dict']['targets'].float().detach(), reduction='none')
        action_loss = action_loss * action_loss_weight[:, None, :, None, None]
        action_loss = action_loss * input_dict['action_dict']['actions_mask'].float()
        # Permute to (B, F, H, W, C) and flatten to (B*F, H*W*C)
        action_loss = action_loss.permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        action_mask = input_dict['action_dict']['actions_mask'].float().permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        action_loss = action_loss.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        action_mask = action_mask.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        # Sum per frame and normalize by mask per frame
        action_loss_per_frame = action_loss.sum(dim=1)  # (B*F,)
        action_mask_per_frame = action_mask.sum(dim=1)  # (B*F,)
        action_loss = (action_loss_per_frame / (action_mask_per_frame + 1e-6)).mean()

        return latent_loss / self.gradient_accumulation_steps, action_loss / self.gradient_accumulation_steps

    def _masked_mean(self, values, mask, dim):
        mask = mask.to(values.dtype)
        while mask.ndim < values.ndim:
            mask = mask.unsqueeze(-1)
        denom = mask.sum(dim=dim).clamp_min(1e-6)
        return (values * mask).sum(dim=dim) / denom

    def _pool_video_delta(self, video_hidden, input_dict):
        latent = input_dict['latent_dict']['latent']
        B, _, F_latent, H_latent, W_latent = latent.shape
        patch_f, patch_h, patch_w = self.patch_size
        F_tokens = F_latent // patch_f
        H_tokens = H_latent // patch_h
        W_tokens = W_latent // patch_w
        expected_tokens = F_tokens * H_tokens * W_tokens
        video_hidden = video_hidden[:, :expected_tokens].float()
        frame_hidden = video_hidden.reshape(B, F_tokens, H_tokens * W_tokens, -1).mean(dim=2)

        delta_mode = getattr(self.config, 'wam_video_delta_mode', 'last_first')
        if F_tokens <= 1 or delta_mode == 'mean_pool':
            return frame_hidden.mean(dim=1)
        if delta_mode == 'mean_step':
            return (frame_hidden[:, 1:] - frame_hidden[:, :-1]).mean(dim=1)
        if delta_mode == 'none':
            return frame_hidden[:, -1]
        return frame_hidden[:, -1] - frame_hidden[:, 0]

    def _action_targets_as_tokens(self, input_dict):
        actions = input_dict['action_dict']['latent'].float()
        action_mask = input_dict['action_dict']['actions_mask'].float()
        action_tokens = rearrange(actions, 'b c f n w -> b (f n w) c')
        action_mask_tokens = rearrange(action_mask, 'b c f n w -> b (f n w) c')
        token_mask = action_mask_tokens.any(dim=-1).float()
        channel_mask = action_mask_tokens.sum(dim=1) > 0
        pooled_action = (action_tokens * action_mask_tokens).sum(dim=1)
        pooled_action = pooled_action / action_mask_tokens.sum(dim=1).clamp_min(1.0)
        return action_tokens, action_mask_tokens, token_mask, channel_mask.float(), pooled_action

    def _relational_loss(self, c_v, c_a):
        if c_v.shape[0] < 2:
            return c_v.sum() * 0.0

        c_v = F.normalize(c_v.float(), dim=-1)
        c_a = F.normalize(c_a.float(), dim=-1)
        sim_v = c_v @ c_v.T
        sim_a = c_a @ c_a.T
        mask = ~torch.eye(c_v.shape[0], dtype=torch.bool, device=c_v.device)
        sim_v = sim_v[mask]
        sim_a = sim_a[mask]
        sim_v = (sim_v - sim_v.mean()) / sim_v.std(unbiased=False).clamp_min(1e-6)
        sim_a = (sim_a - sim_a.mean()) / sim_a.std(unbiased=False).clamp_min(1e-6)
        return F.mse_loss(sim_v, sim_a)

    def _counterfactual_loss(self, c_v, c_a):
        c_v = F.normalize(c_v.float(), dim=-1)
        c_a = F.normalize(c_a.float(), dim=-1)
        score_pos = (c_v * c_a).sum(dim=-1)
        neg_scores = []

        if c_v.shape[0] > 1:
            sim = c_v @ c_a.T
            sim = sim.masked_fill(torch.eye(c_v.shape[0], dtype=torch.bool, device=c_v.device), -float("inf"))
            neg_scores.append(sim.max(dim=1).values)

        if self.wam_action_bank is not None and self.wam_action_bank.numel() > 0:
            bank = F.normalize(self.wam_action_bank.to(c_v.device), dim=-1)
            neg_scores.append((c_v @ bank.T).max(dim=1).values)

        if len(neg_scores) == 0:
            zero = score_pos.sum() * 0.0
            return zero, score_pos.detach().mean(), zero.detach()

        score_neg = torch.stack(neg_scores, dim=0).max(dim=0).values
        tau = max(float(getattr(self.config, 'wam_counterfactual_tau', 0.1)), 1e-6)
        loss = F.softplus((score_neg - score_pos) / tau).mean()
        return loss, score_pos.detach().mean(), score_neg.detach().mean()

    @torch.no_grad()
    def _update_counterfactual_bank(self, c_a):
        bank_size = int(getattr(self.config, 'wam_cf_queue_size', 1024))
        if bank_size <= 0:
            return

        new_codes = c_a.detach().float()
        if dist.is_initialized() and dist.get_world_size() > 1 and getattr(self.config, 'wam_cf_gather_distributed', True):
            gathered = [torch.zeros_like(new_codes) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, new_codes)
            new_codes = torch.cat(gathered, dim=0)

        if self.wam_action_bank is None:
            self.wam_action_bank = new_codes[-bank_size:].detach()
        else:
            self.wam_action_bank = torch.cat([self.wam_action_bank.to(new_codes.device), new_codes], dim=0)
            self.wam_action_bank = self.wam_action_bank[-bank_size:].detach()

    def _compute_wam_losses(self, input_dict, repr_dict):
        video_key = getattr(self.config, 'wam_video_repr_key', 'latent_clean')
        action_key = getattr(self.config, 'wam_action_repr_key', 'action_clean')
        video_delta = self._pool_video_delta(repr_dict[video_key], input_dict)
        action_hidden = repr_dict[action_key].float()
        action_tokens, action_mask_tokens, token_mask, channel_mask, pooled_action = self._action_targets_as_tokens(input_dict)

        seq_len = min(action_hidden.shape[1], action_tokens.shape[1])
        action_hidden = action_hidden[:, :seq_len]
        action_tokens = action_tokens[:, :seq_len]
        action_mask_tokens = action_mask_tokens[:, :seq_len]
        token_mask = token_mask[:, :seq_len]

        adapters = self._adapter_module()
        c_v = adapters.encode_video(video_delta)
        action_token_codes = adapters.encode_action_tokens(action_hidden)
        c_a = self._masked_mean(action_token_codes, token_mask, dim=1)

        scale = 1.0 / self.gradient_accumulation_steps
        raw_losses = {}
        log_items = {}

        if self.wam_stage == 'stage1_adapter':
            action_recon = adapters.decode_action_tokens(action_token_codes)
            recon_loss = F.smooth_l1_loss(action_recon, action_tokens, reduction='none')
            recon_loss = (recon_loss * action_mask_tokens).sum() / action_mask_tokens.sum().clamp_min(1.0)

            inverse_action = adapters.decode_inverse_action(c_v)
            inverse_loss = F.smooth_l1_loss(inverse_action, pooled_action, reduction='none')
            inverse_loss = (inverse_loss * channel_mask).sum() / channel_mask.sum().clamp_min(1.0)

            raw_losses['wam_recon_action_loss'] = recon_loss * float(getattr(self.config, 'wam_recon_action_weight', 1.0))
            raw_losses['wam_inverse_loss'] = inverse_loss * float(getattr(self.config, 'wam_inverse_weight', 1.0))

            relational_weight = float(getattr(self.config, 'wam_relational_weight', 0.0))
            if relational_weight > 0:
                raw_losses['wam_relational_loss'] = self._relational_loss(c_v, c_a) * relational_weight

            cf_weight = float(getattr(self.config, 'wam_counterfactual_weight', 0.1))
            if cf_weight > 0:
                cf_loss, pos_score, neg_score = self._counterfactual_loss(c_v, c_a)
                raw_losses['wam_counterfactual_loss'] = cf_loss * cf_weight
                log_items['wam_cf_pos_score'] = pos_score
                log_items['wam_cf_neg_score'] = neg_score

        elif self.wam_stage == 'stage2_action_posttrain':
            target_c_v = c_v.detach()
            latent_consistency = F.smooth_l1_loss(c_a, target_c_v)
            raw_losses['wam_latent_consistency_loss'] = (
                latent_consistency * float(getattr(self.config, 'wam_latent_consistency_weight', 1.0))
            )

            cf_weight = float(getattr(
                self.config,
                'wam_stage2_counterfactual_weight',
                getattr(self.config, 'wam_counterfactual_weight', 0.0),
            ))
            if cf_weight > 0:
                cf_loss, pos_score, neg_score = self._counterfactual_loss(target_c_v, c_a)
                raw_losses['wam_counterfactual_loss'] = cf_loss * cf_weight
                log_items['wam_cf_pos_score'] = pos_score
                log_items['wam_cf_neg_score'] = neg_score

        total_loss = sum(raw_losses.values()) if len(raw_losses) > 0 else c_a.sum() * 0.0
        scaled_losses = {key: value * scale for key, value in raw_losses.items()}
        scaled_losses['wam_total_loss'] = total_loss * scale
        scaled_losses.update(log_items)
        return scaled_losses, c_a

    def _train_step(self, batch, batch_idx):
        """Train a single batch, returns losses for logging."""
        batch = self.convert_input_format(batch)
        input_dict = self._prepare_input_dict(batch)
        
        should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0
        
        if self.wam_stage != 'stage1_adapter' and hasattr(self.transformer, 'set_requires_gradient_sync'):
            self.transformer.set_requires_gradient_sync(should_sync)

        c_a_for_bank = None
        if self.wam_stage == 'stage1_adapter':
            with torch.no_grad():
                output = self.transformer(input_dict, train_mode=True, return_latents=True)
            wam_losses, c_a_for_bank = self._compute_wam_losses(input_dict, output[2])
            latent_loss = wam_losses['wam_total_loss'].detach() * 0.0
            action_loss = wam_losses['wam_total_loss'].detach() * 0.0
            loss = wam_losses['wam_total_loss']
        elif self.wam_stage == 'stage2_action_posttrain':
            output = self.transformer(input_dict, train_mode=True, return_latents=True)
            latent_loss, action_loss = self.compute_loss(input_dict, output)
            wam_losses, c_a_for_bank = self._compute_wam_losses(input_dict, output[2])
            video_loss_weight = float(getattr(self.config, 'wam_stage2_video_loss_weight', 0.0))
            action_loss_weight = float(getattr(self.config, 'wam_stage2_action_loss_weight', 1.0))
            loss = latent_loss * video_loss_weight + action_loss * action_loss_weight + wam_losses['wam_total_loss']
        else:
            output = self.transformer(input_dict, train_mode=True)
            latent_loss, action_loss = self.compute_loss(input_dict, output)
            wam_losses = {}
            loss = latent_loss + action_loss

        loss.backward()

        losses = {'latent_loss': latent_loss.detach(), 'action_loss': action_loss.detach()}
        for key, value in wam_losses.items():
            losses[key] = value.detach() if torch.is_tensor(value) else value
        
        # Only update weights after accumulating gradients
        if should_sync:
            total_norm = torch.nn.utils.clip_grad_norm_(self.optimized_parameters, 2.0)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            if c_a_for_bank is not None:
                self._update_counterfactual_bank(c_a_for_bank)
            
            losses['total_norm'] = total_norm
            losses['should_log'] = True
        else:
            losses['should_log'] = False

        return losses

    def save_checkpoint(self,):
        """Save model checkpoint in the same format as pretrained model."""
        try:
            save_transformer = not (
                self.wam_stage == 'stage1_adapter' and
                not getattr(self.config, 'wam_save_frozen_transformer', False)
            )
            state_dict_bf16 = None
            if save_transformer:
                state_dict = get_model_state_dict(
                    self.transformer,
                    options=StateDictOptions(full_state_dict=True, cpu_offload=True),
                )
                state_dict_bf16 = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}
            # optim_state = get_optimizer_state_dict(
            #         self.transformer, self.optimizer,
            #         options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            #     )

            # Only rank 0 saves the checkpoint
            if self.config.rank == 0:
                checkpoint_dir = self.save_dir / f"checkpoint_step_{self.step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                if save_transformer:
                    # Save transformer in the same format as pretrained model
                    transformer_dir = checkpoint_dir / "transformer"
                    transformer_dir.mkdir(parents=True, exist_ok=True)

                    logger.info(f"Saving transformer to {transformer_dir}")

                    # Manually save in diffusers format (outside FSDP context to avoid deadlock)
                    # Save model weights
                    model_file = transformer_dir / "diffusion_pytorch_model.safetensors"
                    save_file(state_dict_bf16, model_file)

                    # Save config (copy from original transformer config and update _name_or_path)
                    config_file = transformer_dir / "config.json"
                    config_dict = dict(self.transformer.config)
                    config_dict.pop('_name_or_path', None)
                    with open(config_file, 'w') as f:
                        json.dump(config_dict, f, indent=2)

                if self.wam_adapters is not None:
                    adapters_dir = checkpoint_dir / "adapters"
                    adapters_dir.mkdir(parents=True, exist_ok=True)
                    adapter_file = adapters_dir / "wam_adapters.pt"
                    torch.save({
                        "state_dict": self._adapter_module().state_dict(),
                        "wam_stage": self.wam_stage,
                        "latent_dim": int(getattr(self.config, 'wam_latent_dim', 256)),
                        "video_delta_mode": getattr(self.config, 'wam_video_delta_mode', 'last_first'),
                    }, adapter_file)
                    logger.info(f"Saved WAM adapters to {adapter_file}")

                # # Save optimizer state and training metadata in PyTorch format
                # training_state_path = checkpoint_dir / "training_state.pt"
                # logger.info(f"Saving training state to {training_state_path}")
                # torch.save({
                #     'step': self.step,
                #     'optimizer_state_dict': optim_state,
                #     'config': vars(self.config),
                # }, training_state_path)

                logger.info(f"Checkpoint saved successfully at step {self.step}")

            # Synchronize all processes after saving
            if dist.is_initialized():
                dist.barrier()

        except Exception as e:
            if self.config.rank == 0:
                logger.error(f"Failed to save checkpoint: {e}")
                import traceback
                logger.error(traceback.format_exc())
            # Ensure all processes stay synchronized even on error
            if dist.is_initialized():
                dist.barrier()

    def _load_training_state(self, checkpoint_path):
        """Load training state (optimizer + step) after FSDP and optimizer creation."""
        checkpoint_dir = Path(checkpoint_path)
        training_state_path = checkpoint_dir / "training_state.pt"

        if not training_state_path.exists():
            if self.config.rank == 0:
                logger.warning(f"Training state not found: {training_state_path}, starting from step 0")
            return

        if self.config.rank == 0:
            logger.info(f"Loading training state from {training_state_path}")

        # All ranks load the training state directly
        training_state = torch.load(training_state_path, map_location='cpu', weights_only=False)

        # All ranks load optimizer state (required for FSDP)
        set_optimizer_state_dict(
            self.transformer, self.optimizer,
            optim_state_dict=training_state['optimizer_state_dict'],
            options=StateDictOptions(full_state_dict=True, strict=False)
        )
        self.step = training_state.get('step', 0)

        if self.config.rank == 0:
            logger.info(f"Training state loaded, resuming from step {self.step}")

        # Synchronize all ranks
        if dist.is_initialized():
            dist.barrier()

    def train(self):
        """Main training loop - train by steps instead of epochs."""
        logger.info(f"Starting training for {self.config.num_steps} steps...")
        if self.wam_stage == 'stage1_adapter':
            self.transformer.eval()
            self._adapter_module().train()
        else:
            self.transformer.train()
            if self.wam_adapters is not None:
                self._adapter_module().eval()

        progress_bar = tqdm(
            total=self.config.num_steps,
            desc="Training",
            disable=(self.config.rank != 0),
            leave=True,
            dynamic_ncols=True,
            initial=self.step
        )

        self.optimizer.zero_grad()
        accumulated_latent_losses = []
        accumulated_action_losses = []
        accumulated_extra_items = {}
        step_in_accumulation = 0

        while self.step < self.config.num_steps:
            # Get next batch (handles epoch reset automatically)
            batch = self._get_next_batch()
            
            losses = self._train_step(batch, step_in_accumulation)
            
            # Accumulate losses for logging
            accumulated_latent_losses.append(losses['latent_loss'])
            accumulated_action_losses.append(losses['action_loss'])
            for key, value in losses.items():
                if key in ('latent_loss', 'action_loss', 'total_norm', 'should_log'):
                    continue
                if torch.is_tensor(value):
                    accumulated_extra_items.setdefault(key, []).append(value)
            step_in_accumulation += 1

            # Log and checkpoint when optimizer steps
            if losses['should_log']:
                lr = self.lr_scheduler.get_last_lr()[0]

                # Average accumulated losses
                latent_loss_show = dist_mean(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                action_loss_show = dist_mean(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()
                max_latent_loss_show = dist_max(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                max_action_loss_show = dist_max(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()
                extra_show = {}
                for key, values in accumulated_extra_items.items():
                    stacked = torch.stack(values)
                    aggregate = stacked.sum() if key.endswith('_loss') else stacked.mean()
                    extra_show[key] = dist_mean(aggregate).detach().cpu().item()

                # Clear accumulated losses
                accumulated_latent_losses = []
                accumulated_action_losses = []
                accumulated_extra_items = {}
                step_in_accumulation = 0

                torch.cuda.synchronize()
                if self.step % self.config.gc_interval == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                if self.config.rank == 0:
                    total_norm = losses['total_norm']
                    progress_bar.n += 1
                    postfix = {
                        'latent_loss': f'{latent_loss_show:.4f}',
                        'action_loss': f'{action_loss_show:.4f}',
                        'step': self.step,
                        'grad_norm': f'{total_norm.item():.2f}',
                        'lr': f'{lr:.2e}'
                    }
                    if 'wam_total_loss' in extra_show:
                        postfix['wam_loss'] = f"{extra_show['wam_total_loss']:.4f}"
                    progress_bar.set_postfix(postfix)
                    if self.config.enable_wandb:
                        wandb_log = {
                            'loss_metrics/global_avg_video_loss': latent_loss_show,
                            'loss_metrics/global_avg_action_loss': action_loss_show,
                            'loss_metrics/global_max_video_loss': max_latent_loss_show,
                            'loss_metrics/global_max_action_loss': max_action_loss_show,
                            'grad_norm': total_norm.item(),
                            'lr': lr,
                        }
                        for key, value in extra_show.items():
                            metric_group = 'loss_metrics' if key.endswith('_loss') else 'wam_metrics'
                            wandb_log[f'{metric_group}/{key}'] = value
                        self.wandb.log(wandb_log, step=self.step)
                
                self.step += 1
                
                if self.step % self.config.save_interval == 0:
                    if self.config.rank == 0:
                        logger.info(f"Starting save model at step {self.step}")
                    self.save_checkpoint()

            if dist.is_initialized():
                dist.barrier()

        progress_bar.close()
        logger.info("Training completed!")


def run(args):
    """Main entry point."""
    config = VA_CONFIGS[args.config_name]

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    init_distributed(world_size, local_rank, rank)

    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size

    if args.save_root is not None:
        config.save_root = args.save_root
    if args.resume_from is not None:
        config.resume_from = args.resume_from
    if args.wam_posttrain_stage is not None:
        config.wam_posttrain_stage = args.wam_posttrain_stage
    if args.wam_adapter_path is not None:
        config.wam_adapter_path = args.wam_adapter_path
    if args.disable_wandb:
        config.enable_wandb = False

    if rank == 0:
        logger.info(f"Using config: {args.config_name}")
        logger.info(f"World size: {world_size}, Local rank: {local_rank}")
        logger.info(f"WAM posttraining stage: {getattr(config, 'wam_posttrain_stage', 'sft')}")

    trainer = Trainer(config)
    trainer.train()


def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(description="Train WAN model for robotics")
    parser.add_argument(
        "--config-name",
        type=str,
        default='robotwin_train',
        help="Config name",
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default=None,
        help="Root directory for saving checkpoints",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Checkpoint directory to resume transformer/adapters from",
    )
    parser.add_argument(
        "--wam-posttrain-stage",
        type=str,
        default=None,
        choices=['sft', 'stage1_adapter', 'stage2_action_posttrain'],
        help="WAM posttraining mode",
    )
    parser.add_argument(
        "--wam-adapter-path",
        type=str,
        default=None,
        help="Path to adapters/wam_adapters.pt for Stage 2 or adapter resume",
    )
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Disable WandB logging regardless of config",
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    init_logger()
    main()
