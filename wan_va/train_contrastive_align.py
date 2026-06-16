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
   representations in a single forward pass, without touching ``model.py``.
2. Video residual tokens are formed per frame as ``v[t+1] - v[t]`` (mean-pooled
   over spatial tokens). Action tokens are mean-pooled over the per-frame action
   tokens with the action mask. They are paired frame-wise: residual[t] <-> a_t.
3. Two small MLP adapters project both into a shared subspace, and we apply a
   symmetric (CLIP-style) InfoNCE with in-batch negatives and a learnable
   temperature.
"""
import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS
from distributed.util import init_distributed
from modules.latent_posttraining import MLP
from train import Trainer
from utils import init_logger, logger, warmup_constant_lambda


class ContrastiveAlignHead(nn.Module):
    """Projects video-residual and action tokens into a shared subspace and
    computes a symmetric InfoNCE alignment loss with a learnable temperature."""

    def __init__(
        self,
        hidden_dim,
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
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(float(logit_scale_init))))
        self.logit_scale_max = float(logit_scale_max)

    def forward(self, video_residual, action_feat):
        # video_residual / action_feat: [N, hidden_dim] (already valid-filtered)
        c_v = F.normalize(self.video_adapter(video_residual.float()), dim=-1)
        c_a = F.normalize(self.action_adapter(action_feat.float()), dim=-1)

        scale = self.logit_scale.exp().clamp(max=self.logit_scale_max)
        logits = scale * c_v @ c_a.t()  # [N, N]
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_v2a = F.cross_entropy(logits, labels)
        loss_a2v = F.cross_entropy(logits.t(), labels)
        loss = 0.5 * (loss_v2a + loss_a2v)

        with torch.no_grad():
            top1 = (logits.argmax(dim=-1) == labels).float().mean()
        return loss, top1


class ContrastiveAlignTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)

        self.lambda_align = float(getattr(config, "lambda_align", 0.1))
        # residual[t] = v[t+1]-v[t] is paired with action frame (t + action_frame_offset)
        self.action_frame_offset = int(getattr(config, "align_action_frame_offset", 0))

        hidden_dim = self.transformer.num_attention_heads * self.transformer.attention_head_dim
        self.align_head = ContrastiveAlignHead(
            hidden_dim=hidden_dim,
            subspace_dim=getattr(config, "align_subspace_dim", 256),
            adapter_hidden_dim=getattr(config, "align_adapter_hidden_dim", 1024),
            adapter_depth=getattr(config, "align_adapter_depth", 2),
            dropout=getattr(config, "align_adapter_dropout", 0.0),
            logit_scale_init=getattr(config, "align_logit_scale_init", 1.0 / 0.07),
        ).to(self.device, dtype=torch.float32)

        # The head is replicated (not FSDP-sharded). Keep it identical across
        # ranks and average its grads manually at each optimizer step.
        self._broadcast_head()

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

        # Capture buffers + hooks on the projection layers' inputs.
        self._captured = {}
        self.transformer.proj_out.register_forward_pre_hook(self._make_hook("video"))
        self.transformer.action_proj_out.register_forward_pre_hook(self._make_hook("action"))

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
            return zero, zero.detach()

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
        amask = input_dict["action_dict"]["actions_mask"].any(dim=1).squeeze(-1).bool()
        tok_mask = amask.unsqueeze(-1).float()  # [B, F, n, 1]
        act = (act * tok_mask).sum(dim=2) / tok_mask.sum(dim=2).clamp_min(1e-6)  # [B, F, C]
        frame_valid = amask.any(dim=2)  # [B, F]

        if Fn < 2:
            zero = torch.zeros((), device=self.device)
            return zero, zero.detach()

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

        residual = residual.reshape(-1, C)
        act_use = act_use.reshape(-1, C)
        valid = valid.reshape(-1)

        residual = residual[valid]
        act_use = act_use[valid]

        if residual.shape[0] < 2:
            zero = torch.zeros((), device=self.device)
            return zero, zero.detach()

        return self.align_head(residual, act_use)

    # ---- training step (mirrors parent, adds alignment loss) ----------------
    def _train_step(self, batch, batch_idx):
        batch = self.convert_input_format(batch)
        input_dict = self._prepare_input_dict(batch)

        should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0
        self.transformer.set_requires_gradient_sync(should_sync)

        self._captured = {}
        output = self.transformer(input_dict, train_mode=True)
        latent_loss, action_loss = self.compute_loss(input_dict, output)
        align_loss, align_acc = self._compute_align_loss(input_dict)

        loss = latent_loss + action_loss + (
            self.lambda_align * align_loss / self.gradient_accumulation_steps
        )
        loss.backward()

        losses = {
            "latent_loss": latent_loss.detach(),
            "action_loss": action_loss.detach(),
            "align_loss": align_loss.detach(),
            "align_acc": align_acc.detach(),
        }

        if should_sync:
            self._sync_head_grads()
            total_norm = torch.nn.utils.clip_grad_norm_(
                list(self.transformer.parameters()) + list(self.align_head.parameters()),
                2.0,
            )
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

            losses["total_norm"] = total_norm
            losses["should_log"] = True
        else:
            losses["should_log"] = False

        return losses


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
    if args.lambda_align is not None:
        config.lambda_align = args.lambda_align

    if rank == 0:
        logger.info(f"Using config: {args.config_name}")
        logger.info(f"World size: {world_size}, Local rank: {local_rank}")
        logger.info(f"lambda_align: {getattr(config, 'lambda_align', 0.1)}")

    trainer = ContrastiveAlignTrainer(config)
    trainer.train()


def main():
    parser = argparse.ArgumentParser(
        description="Train WAN model for robotics with contrastive video/action alignment"
    )
    parser.add_argument("--config-name", type=str, default="robotwin_train", help="Config name")
    parser.add_argument("--save-root", type=str, default=None, help="Root directory for checkpoints")
    parser.add_argument("--lambda-align", type=float, default=None, help="Weight of the alignment loss")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    init_logger()
    main()
