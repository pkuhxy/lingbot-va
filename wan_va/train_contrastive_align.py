# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Joint training with a per-frame contrastive alignment between the video
residual representation (s_{t+1} - s_t) and the action representation (a_t).

This file is intentionally standalone: it subclasses the original ``Trainer``
defined in ``train.py`` and does NOT modify it. The alignment signal is injected
on top of the existing flow-matching MSE losses so that it actually shapes the
MMDiT internal representations (the backbone is trained, not frozen).

Design
------
1. We grab the final per-token hidden states of both branches (the inputs to
   ``proj_out`` / ``action_proj_out``) via forward-pre-hooks during the normal
   ``forward_train`` call. This gives us both the MSE predictions and the
   denoising-path representations in a single forward pass, without touching
   ``model.py``. We intentionally align the noisy prediction branch instead of
   a fully clean branch, so the shared space is trained under the same denoising
   perturbations that the backbone actually sees.
2. Video residual tokens are formed per frame as ``v[t+1] - v[t]`` (mean-pooled
   over spatial tokens). Action tokens are mean-pooled over the per-frame action
   tokens with the action mask. They are paired frame-wise: residual[t] <-> a_t.
3. Two small MLP adapters project both into a shared subspace, and we apply a
   symmetric (CLIP-style) InfoNCE with in-batch negatives and a learnable
   temperature. A lightweight inverse decoder also reconstructs the paired
   action from the video-residual subspace, discouraging representation drift.
"""
import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    StateDictOptions,
)
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS
from distributed.util import dist_max, dist_mean, init_distributed
from modules.latent_posttraining import MLP, masked_smooth_l1
from train import Trainer
from utils import init_logger, logger, warmup_constant_lambda


class ContrastiveAlignHead(nn.Module):
    """Projects video-residual and action tokens into a shared subspace and
    computes a symmetric InfoNCE alignment loss with a learnable temperature."""

    def __init__(
        self,
        hidden_dim,
        action_dim,
        subspace_dim=256,
        adapter_hidden_dim=1024,
        adapter_depth=2,
        dropout=0.0,
        logit_scale_init=1.0 / 0.07,
        logit_scale_max=100.0,
    ):
        super().__init__()
        self.video_adapter = MLP(
            hidden_dim, subspace_dim,
            hidden_dim=adapter_hidden_dim, depth=adapter_depth, dropout=dropout,
        )
        self.action_adapter = MLP(
            hidden_dim, subspace_dim,
            hidden_dim=adapter_hidden_dim, depth=adapter_depth, dropout=dropout,
        )
        self.inverse_decoder = MLP(
            subspace_dim, action_dim,
            hidden_dim=adapter_hidden_dim, depth=adapter_depth, dropout=dropout,
        )
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(float(logit_scale_init))))
        self.logit_scale_max = float(logit_scale_max)

    def forward(self, video_residual, action_feat, action_target=None, action_target_mask=None):
        # video_residual / action_feat: [N, hidden_dim] (already valid-filtered)
        z_v = self.video_adapter(video_residual.float())
        z_a = self.action_adapter(action_feat.float())
        c_v = F.normalize(z_v, dim=-1)
        c_a = F.normalize(z_a, dim=-1)

        scale = self.logit_scale.exp().clamp(max=self.logit_scale_max)
        logits = scale * c_v @ c_a.t()  # [N, N]
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_v2a = F.cross_entropy(logits, labels)
        loss_a2v = F.cross_entropy(logits.t(), labels)
        align_loss = 0.5 * (loss_v2a + loss_a2v)

        if action_target is None:
            recon_loss = video_residual.new_zeros(())
        else:
            action_recon = self.inverse_decoder(z_v)
            recon_loss = masked_smooth_l1(
                action_recon,
                action_target.float(),
                action_target_mask,
                beta=0.1,
            )

        with torch.no_grad():
            top1 = (logits.argmax(dim=-1) == labels).float().mean()
        return align_loss, top1, recon_loss


class ContrastiveAlignTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)

        self.lambda_align = float(getattr(config, "lambda_align", 0.1))
        self.lambda_action_recon = float(getattr(config, "lambda_action_recon", 0.1))
        # residual[t] = v[t+1]-v[t] is paired with action frame (t + action_frame_offset)
        self.action_frame_offset = int(getattr(config, "align_action_frame_offset", 0))

        hidden_dim = self.transformer.num_attention_heads * self.transformer.attention_head_dim
        action_dim = getattr(config, "action_dim", getattr(self.transformer.config, "action_dim", 30))
        self.align_head = ContrastiveAlignHead(
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            subspace_dim=getattr(config, "align_subspace_dim", 256),
            adapter_hidden_dim=getattr(config, "align_adapter_hidden_dim", 1024),
            adapter_depth=getattr(config, "align_adapter_depth", 2),
            dropout=getattr(config, "align_adapter_dropout", 0.0),
            logit_scale_init=getattr(config, "align_logit_scale_init", 1.0 / 0.07),
        ).to(self.device, dtype=torch.float32)

        # Register the head params with the existing optimizer, then rebuild the
        # LambdaLR so its per-group lambdas match the new param-group count.
        self.optimizer.add_param_group({
            "params": [p for p in self.align_head.parameters() if p.requires_grad],
            "lr": getattr(config, "align_learning_rate", config.learning_rate),
            "weight_decay": getattr(config, "align_weight_decay", config.weight_decay),
        })
        warmup_steps = config.warmup_steps
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=[
                (lambda step: warmup_constant_lambda(step, warmup_steps=warmup_steps))
                for _ in self.optimizer.param_groups
            ],
        )

        self._load_contrastive_checkpoint(getattr(config, "resume_from", None))

        # The head is replicated (not FSDP-sharded). Keep it identical across
        # ranks and average its grads manually at each optimizer step.
        self._broadcast_head()

        # Capture buffers + hooks on the projection layers' inputs.
        self._captured = {}
        self.transformer.proj_out.register_forward_pre_hook(self._make_hook("video"))
        self.transformer.action_proj_out.register_forward_pre_hook(self._make_hook("action"))

    # ---- checkpoint helpers -------------------------------------------------
    def _align_head_config(self):
        return {
            "hidden_dim": self.transformer.num_attention_heads * self.transformer.attention_head_dim,
            "action_dim": getattr(self.config, "action_dim", getattr(self.transformer.config, "action_dim", 30)),
            "subspace_dim": getattr(self.config, "align_subspace_dim", 256),
            "adapter_hidden_dim": getattr(self.config, "align_adapter_hidden_dim", 1024),
            "adapter_depth": getattr(self.config, "align_adapter_depth", 2),
            "adapter_dropout": getattr(self.config, "align_adapter_dropout", 0.0),
            "logit_scale_init": getattr(self.config, "align_logit_scale_init", 1.0 / 0.07),
            "logit_scale_max": self.align_head.logit_scale_max,
            "lambda_align": self.lambda_align,
            "lambda_action_recon": self.lambda_action_recon,
            "align_action_frame_offset": self.action_frame_offset,
            "align_learning_rate": getattr(self.config, "align_learning_rate", self.config.learning_rate),
            "align_weight_decay": getattr(self.config, "align_weight_decay", self.config.weight_decay),
        }

    def _save_align_head(self, checkpoint_dir):
        align_dir = checkpoint_dir / "align_head"
        align_dir.mkdir(parents=True, exist_ok=True)

        align_state = {
            key: value.detach().float().cpu().contiguous()
            for key, value in self.align_head.state_dict().items()
        }
        save_file(align_state, align_dir / "model.safetensors")

        with open(align_dir / "config.json", "w") as f:
            json.dump(self._align_head_config(), f, indent=2)

    def _load_align_head(self, checkpoint_dir):
        align_file = checkpoint_dir / "align_head" / "model.safetensors"
        if not align_file.exists():
            if self.config.rank == 0:
                logger.warning(f"Align head checkpoint not found: {align_file}; using a fresh head")
            return

        state_dict = load_file(align_file, device="cpu")
        load_result = self.align_head.load_state_dict(state_dict, strict=False)
        if self.config.rank == 0:
            logger.info(f"Loaded align head from {align_file}")
            if load_result.missing_keys:
                logger.warning(f"Missing align head keys while loading: {load_result.missing_keys}")
            if load_result.unexpected_keys:
                logger.warning(f"Unexpected align head keys while loading: {load_result.unexpected_keys}")

    def _load_contrastive_state(self, checkpoint_dir):
        trainer_state_path = checkpoint_dir / "trainer_state.pt"
        if trainer_state_path.exists():
            state = torch.load(trainer_state_path, map_location="cpu", weights_only=False)
            self.step = int(state.get("step", self.step))
            scheduler_state = state.get("lr_scheduler_state_dict")
            if scheduler_state is not None:
                self.lr_scheduler.load_state_dict(scheduler_state)
            if self.config.rank == 0:
                logger.info(f"Loaded contrastive trainer state from {trainer_state_path}")
            return

        prefix = "checkpoint_step_"
        if checkpoint_dir.name.startswith(prefix):
            try:
                self.step = int(checkpoint_dir.name[len(prefix):])
                if self.config.rank == 0:
                    logger.info(f"Inferred resume step {self.step} from {checkpoint_dir.name}")
            except ValueError:
                pass

    def _load_contrastive_checkpoint(self, checkpoint_path):
        if not checkpoint_path:
            return

        checkpoint_dir = Path(checkpoint_path)
        self._load_align_head(checkpoint_dir)
        self._load_contrastive_state(checkpoint_dir)

    # ---- head distributed helpers ------------------------------------------
    def _broadcast_head(self):
        if dist.is_initialized():
            for p in self.align_head.parameters():
                dist.broadcast(p.data, src=0)

    def _sync_head_grads(self):
        if not dist.is_initialized():
            return
        for p in self.align_head.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

    @staticmethod
    def _metric_item(value):
        if hasattr(value, "to_local"):
            value = value.to_local()
        if torch.is_tensor(value):
            return value.detach().float().cpu().item()
        return float(value)

    # ---- representation capture --------------------------------------------
    def _make_hook(self, name):
        def hook(module, args):
            # args[0]: [1, B*L, C] hidden states fed into the projection layer
            self._captured[name] = args[0]
            return None
        return hook

    def _compute_align_loss(self, input_dict):
        if "video" not in self._captured or "action" not in self._captured:
            zero = torch.zeros((), device=self.device)
            return zero, zero.detach(), zero

        targets = input_dict["latent_dict"]["targets"]
        B, _, Fn = targets.shape[0], targets.shape[1], targets.shape[2]
        C = self._captured["video"].shape[-1]

        # [1, B*F*HW, C] -> [B, F, HW, C] -> mean over spatial -> [B, F, C]
        vid = self._captured["video"].reshape(B, Fn, -1, C).float()
        vid = vid.mean(dim=2)
        # [1, B*F*n, C] -> [B, F, n, C]
        act = self._captured["action"].reshape(B, Fn, -1, C).float()

        # per-(frame, action-token) validity from the action mask
        # actions_mask: [B, 30, F, n, 1] -> token mask [B, F, n]
        action_mask = input_dict["action_dict"]["actions_mask"].bool()
        amask = action_mask.any(dim=1).squeeze(-1).bool()
        tok_mask = amask.unsqueeze(-1).float()  # [B, F, n, 1]
        act = (act * tok_mask).sum(dim=2) / tok_mask.sum(dim=2).clamp_min(1e-6)  # [B, F, C]
        frame_valid = amask.any(dim=2)  # [B, F]

        clean_action = input_dict["action_dict"]["latent"].float().squeeze(-1).permute(0, 2, 3, 1)
        clean_action_mask = action_mask.squeeze(-1).permute(0, 2, 3, 1)
        clean_mask_f = clean_action_mask.float()
        action_target = (
            clean_action * clean_mask_f
        ).sum(dim=2) / clean_mask_f.sum(dim=2).clamp_min(1e-6)  # [B, F, action_dim]
        action_target_mask = clean_action_mask.any(dim=2)  # [B, F, action_dim]

        if Fn < 2:
            zero = torch.zeros((), device=self.device)
            return zero, zero.detach(), zero

        # residual[t] = v[t+1] - v[t], for t in [0, F-2]
        residual = vid[:, 1:] - vid[:, :-1]  # [B, F-1, C]

        off = self.action_frame_offset
        # action frame index paired with residual[t] is (t + off), clamp range
        act_start = max(off, 0)
        act_end = act_start + (Fn - 1)
        if act_end > Fn:
            # not enough action frames for this offset; fall back to offset 0
            act_start, act_end = 0, Fn - 1
        act_use = act[:, act_start:act_end]            # [B, F-1, C]
        valid = frame_valid[:, act_start:act_end]      # [B, F-1]
        action_target = action_target[:, act_start:act_end]
        action_target_mask = action_target_mask[:, act_start:act_end]

        residual = residual.reshape(-1, C)
        act_use = act_use.reshape(-1, C)
        valid = valid.reshape(-1)
        action_dim = action_target.shape[-1]
        action_target = action_target.reshape(-1, action_dim)
        action_target_mask = action_target_mask.reshape(-1, action_dim)

        residual = residual[valid]
        act_use = act_use[valid]
        action_target = action_target[valid]
        action_target_mask = action_target_mask[valid]

        if residual.shape[0] < 2:
            zero = torch.zeros((), device=self.device)
            return zero, zero.detach(), zero

        return self.align_head(residual, act_use, action_target, action_target_mask)

    # ---- training step (mirrors parent, adds alignment loss) ----------------
    def _train_step(self, batch, batch_idx):
        batch = self.convert_input_format(batch)
        input_dict = self._prepare_input_dict(batch)

        should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0
        self.transformer.set_requires_gradient_sync(should_sync)

        self._captured = {}
        output = self.transformer(input_dict, train_mode=True)
        latent_loss, action_loss = self.compute_loss(input_dict, output)
        align_loss, align_acc, action_recon_loss = self._compute_align_loss(input_dict)

        loss = latent_loss + action_loss + (
            self.lambda_align * align_loss / self.gradient_accumulation_steps
        ) + (
            self.lambda_action_recon * action_recon_loss / self.gradient_accumulation_steps
        )
        loss.backward()

        losses = {
            "latent_loss": latent_loss.detach(),
            "action_loss": action_loss.detach(),
            "align_loss": align_loss.detach(),
            "align_acc": align_acc.detach(),
            "action_recon_loss": action_recon_loss.detach(),
        }

        if should_sync:
            self._sync_head_grads()
            # Transformer params are FSDP/DTensor-backed, while align_head is a
            # normal replicated module. Clip separately because foreach norm
            # cannot mix Tensor and DTensor inputs.
            total_norm = torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 2.0)
            align_total_norm = torch.nn.utils.clip_grad_norm_(
                self.align_head.parameters(),
                2.0,
                foreach=False,
            )
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

            losses["total_norm"] = total_norm
            losses["align_total_norm"] = align_total_norm
            losses["should_log"] = True
        else:
            losses["should_log"] = False

        return losses

    def save_checkpoint(self):
        """Save transformer plus contrastive alignment artifacts."""
        try:
            state_dict = get_model_state_dict(
                self.transformer,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
            state_dict_bf16 = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}

            if self.config.rank == 0:
                checkpoint_dir = self.save_dir / f"checkpoint_step_{self.step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                transformer_dir = checkpoint_dir / "transformer"
                transformer_dir.mkdir(parents=True, exist_ok=True)

                logger.info(f"Saving transformer to {transformer_dir}")
                save_file(state_dict_bf16, transformer_dir / "diffusion_pytorch_model.safetensors")

                config_dict = dict(self.transformer.config)
                config_dict.pop("_name_or_path", None)
                with open(transformer_dir / "config.json", "w") as f:
                    json.dump(config_dict, f, indent=2)

                logger.info(f"Saving align head to {checkpoint_dir / 'align_head'}")
                self._save_align_head(checkpoint_dir)

                torch.save(
                    {
                        "step": self.step,
                        "lambda_align": self.lambda_align,
                        "lambda_action_recon": self.lambda_action_recon,
                        "action_frame_offset": self.action_frame_offset,
                        "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
                    },
                    checkpoint_dir / "trainer_state.pt",
                )

                logger.info(f"Contrastive checkpoint saved successfully at step {self.step}")

            if dist.is_initialized():
                dist.barrier()

        except Exception as e:
            if self.config.rank == 0:
                logger.error(f"Failed to save contrastive checkpoint: {e}")
                import traceback
                logger.error(traceback.format_exc())
            if dist.is_initialized():
                dist.barrier()

    def train(self):
        """Main training loop with contrastive alignment metrics."""
        logger.info(f"Starting contrastive alignment training for {self.config.num_steps} steps...")
        self.transformer.train()
        self.align_head.train()

        progress_bar = tqdm(
            total=self.config.num_steps,
            desc="Contrastive training",
            disable=(self.config.rank != 0),
            leave=True,
            dynamic_ncols=True,
            initial=self.step,
        )

        self.optimizer.zero_grad()
        accumulated_latent_losses = []
        accumulated_action_losses = []
        accumulated_align_losses = []
        accumulated_align_accs = []
        accumulated_action_recon_losses = []
        step_in_accumulation = 0

        while self.step < self.config.num_steps:
            batch = self._get_next_batch()
            losses = self._train_step(batch, step_in_accumulation)

            accumulated_latent_losses.append(losses["latent_loss"])
            accumulated_action_losses.append(losses["action_loss"])
            accumulated_align_losses.append(losses["align_loss"])
            accumulated_align_accs.append(losses["align_acc"])
            accumulated_action_recon_losses.append(losses["action_recon_loss"])
            step_in_accumulation += 1

            if losses["should_log"]:
                lr = self.lr_scheduler.get_last_lr()[0]
                align_lr = self.lr_scheduler.get_last_lr()[-1]

                latent_loss_show = dist_mean(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                action_loss_show = dist_mean(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()
                align_loss_show = dist_mean(torch.stack(accumulated_align_losses).mean()).detach().cpu().item()
                align_acc_show = dist_mean(torch.stack(accumulated_align_accs).mean()).detach().cpu().item()
                action_recon_loss_show = dist_mean(
                    torch.stack(accumulated_action_recon_losses).mean()
                ).detach().cpu().item()
                max_latent_loss_show = dist_max(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                max_action_loss_show = dist_max(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()
                max_align_loss_show = dist_max(torch.stack(accumulated_align_losses).mean()).detach().cpu().item()
                max_action_recon_loss_show = dist_max(
                    torch.stack(accumulated_action_recon_losses).mean()
                ).detach().cpu().item()
                weighted_align_loss = self.lambda_align * align_loss_show
                weighted_action_recon_loss = self.lambda_action_recon * action_recon_loss_show
                total_loss_show = (
                    latent_loss_show
                    + action_loss_show
                    + weighted_align_loss
                    + weighted_action_recon_loss
                )

                accumulated_latent_losses = []
                accumulated_action_losses = []
                accumulated_align_losses = []
                accumulated_align_accs = []
                accumulated_action_recon_losses = []
                step_in_accumulation = 0

                torch.cuda.synchronize()
                if self.step % self.config.gc_interval == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                if self.config.rank == 0:
                    total_norm = self._metric_item(losses["total_norm"])
                    align_total_norm = self._metric_item(losses["align_total_norm"])
                    logit_scale = self.align_head.logit_scale.exp().clamp(
                        max=self.align_head.logit_scale_max
                    ).detach().cpu().item()
                    progress_bar.n += 1
                    progress_bar.set_postfix({
                        "latent_loss": f"{latent_loss_show:.4f}",
                        "action_loss": f"{action_loss_show:.4f}",
                        "align_loss": f"{align_loss_show:.4f}",
                        "act_recon": f"{action_recon_loss_show:.4f}",
                        "align_acc": f"{align_acc_show:.3f}",
                        "total": f"{total_loss_show:.4f}",
                        "step": self.step,
                        "grad_norm": f"{total_norm:.2f}",
                        "align_grad": f"{align_total_norm:.2f}",
                        "lr": f"{lr:.2e}",
                    })
                    if self.config.enable_wandb:
                        self.wandb.log({
                            "loss_metrics/global_avg_video_loss": latent_loss_show,
                            "loss_metrics/global_avg_action_loss": action_loss_show,
                            "loss_metrics/global_avg_align_loss": align_loss_show,
                            "loss_metrics/global_avg_action_recon_loss": action_recon_loss_show,
                            "loss_metrics/global_max_video_loss": max_latent_loss_show,
                            "loss_metrics/global_max_action_loss": max_action_loss_show,
                            "loss_metrics/global_max_align_loss": max_align_loss_show,
                            "loss_metrics/global_max_action_recon_loss": max_action_recon_loss_show,
                            "loss_metrics/weighted_align_loss": weighted_align_loss,
                            "loss_metrics/weighted_action_recon_loss": weighted_action_recon_loss,
                            "loss_metrics/total_loss": total_loss_show,
                            "align_metrics/top1": align_acc_show,
                            "align_metrics/logit_scale": logit_scale,
                            "grad_norm": total_norm,
                            "align_grad_norm": align_total_norm,
                            "lr": lr,
                            "align_lr": align_lr,
                        }, step=self.step)

                self.step += 1

                if self.step % self.config.save_interval == 0:
                    if self.config.rank == 0:
                        logger.info(f"Starting save model at step {self.step}")
                    self.save_checkpoint()

            if dist.is_initialized():
                dist.barrier()

        progress_bar.close()
        logger.info("Contrastive alignment training completed!")


def run(args):
    config = VA_CONFIGS[args.config_name]

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    init_distributed(world_size, local_rank, rank)

    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size

    if args.save_root is not None:
        config.save_root = args.save_root
    if args.resume_from is not None:
        config.resume_from = args.resume_from
    if args.lambda_align is not None:
        config.lambda_align = args.lambda_align
    if args.lambda_action_recon is not None:
        config.lambda_action_recon = args.lambda_action_recon
    if args.enable_wandb and args.disable_wandb:
        raise ValueError("--enable-wandb and --disable-wandb cannot be used together")
    if args.enable_wandb:
        config.enable_wandb = True
    if args.disable_wandb:
        config.enable_wandb = False

    if rank == 0:
        logger.info(f"Using config: {args.config_name}")
        logger.info(f"World size: {world_size}, Local rank: {local_rank}")
        logger.info(f"Per-GPU batch size: {config.batch_size}")
        logger.info(f"Training latent frames: {getattr(config, 'train_frame_num', 'full')}")
        logger.info(
            f"Effective global batch size: "
            f"{config.batch_size * world_size * getattr(config, 'gradient_accumulation_steps', 1)}"
        )
        logger.info(f"lambda_align: {getattr(config, 'lambda_align', 0.1)}")
        logger.info(f"lambda_action_recon: {getattr(config, 'lambda_action_recon', 0.1)}")
        logger.info(f"enable_wandb: {getattr(config, 'enable_wandb', False)}")

    trainer = ContrastiveAlignTrainer(config)
    trainer.train()


def main():
    parser = argparse.ArgumentParser(
        description="Train WAN model for robotics with contrastive video/action alignment"
    )
    parser.add_argument("--config-name", type=str, default="robotwin_contrastive_align", help="Config name")
    parser.add_argument("--save-root", type=str, default=None, help="Root directory for checkpoints")
    parser.add_argument("--resume-from", type=str, default=None, help="Checkpoint directory to resume from")
    parser.add_argument("--enable-wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--disable-wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--lambda-align", type=float, default=None, help="Weight of the alignment loss")
    parser.add_argument(
        "--lambda-action-recon",
        type=float,
        default=None,
        help="Weight of the video-residual-to-action reconstruction loss",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    init_logger()
    main()
