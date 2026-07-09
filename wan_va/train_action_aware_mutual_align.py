"""Mutual action-aware transition alignment for joint video/action flow matching.

This ablation keeps the action-aware ingredients that were introduced after the
first hard contrastive baseline:

1. denoised/predicted-clean video transitions x0[t+1] - x0[t];
2. full action-chunk supervision;
3. action-similarity soft positives; and
4. auxiliary warmup plus a primary-loss-relative cap.

The key difference from ``train_action_aware_align.py`` is that the alignment
term is no longer one-way video -> detached action vectors.  We additionally
read the action branch hidden state during the normal transformer forward pass
and align video-transition embeddings with action-branch embeddings using a
symmetric soft contrastive loss.  Both the video and action branches therefore
receive auxiliary alignment gradients, while the full action chunk is still used
as the detached semantic signal that defines soft positives and the lightweight
video-transition probe.
"""

import argparse
import copy
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
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS
from distributed.util import dist_max, dist_mean, init_distributed
from modules.latent_posttraining import MLP, masked_smooth_l1
from train import Trainer
from utils import data_seq_to_patch, init_logger, logger, warmup_constant_lambda


class ActionAwareMutualTransitionHead(nn.Module):
    """Embeds denoised video transitions and action-branch states jointly."""

    def __init__(
        self,
        video_feature_dim,
        action_feature_dim,
        action_vector_dim,
        subspace_dim=512,
        hidden_dim=1024,
        depth=3,
        dropout=0.0,
    ):
        super().__init__()
        self.video_encoder = MLP(
            video_feature_dim,
            subspace_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
        )
        self.action_encoder = MLP(
            action_feature_dim,
            subspace_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
        )
        self.action_decoder = MLP(
            subspace_dim,
            action_vector_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
        )

    def forward(self, video_features, action_features):
        video_embedding = self.video_encoder(video_features.float())
        action_embedding = self.action_encoder(action_features.float())
        action_prediction = self.action_decoder(video_embedding)
        return video_embedding, action_embedding, action_prediction

class ActionAwareAlignTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)

        self.lambda_action_chunk = float(getattr(config, "lambda_action_chunk", 0.1))
        self.lambda_soft_contrastive = float(
            getattr(config, "lambda_soft_contrastive", 0.05)
        )
        self.aux_warmup_steps = int(getattr(config, "aux_warmup_steps", 250))
        self.aux_max_primary_fraction = float(
            getattr(config, "aux_max_primary_fraction", 0.15)
        )
        self.soft_positive_mix = float(getattr(config, "soft_positive_mix", 0.5))
        self.teacher_temperature = float(getattr(config, "teacher_temperature", 0.1))
        self.student_temperature = float(getattr(config, "student_temperature", 0.1))
        self.transition_pool_size = int(getattr(config, "transition_pool_size", 4))
        self.transition_energy_pool_size = int(
            getattr(config, "transition_energy_pool_size", 2)
        )
        self.action_frame_offset = int(getattr(config, "align_action_frame_offset", 0))

        latent_channels = int(self.transformer.config.out_channels)
        action_dim = int(getattr(config, "action_dim", self.transformer.config.action_dim))
        action_tokens = int(getattr(config, "action_per_frame", 16))
        self.action_dim = action_dim
        self.action_tokens = action_tokens
        video_feature_dim = latent_channels * (
            self.transition_pool_size ** 2 + self.transition_energy_pool_size ** 2
        )
        action_feature_dim = (
            self.transformer.num_attention_heads * self.transformer.attention_head_dim
        )
        self.action_feature_dim = action_feature_dim

        self.action_aware_head = ActionAwareMutualTransitionHead(
            video_feature_dim=video_feature_dim,
            action_feature_dim=action_feature_dim,
            action_vector_dim=action_dim * action_tokens,
            subspace_dim=int(getattr(config, "action_aware_subspace_dim", 512)),
            hidden_dim=int(getattr(config, "action_aware_hidden_dim", 1024)),
            depth=int(getattr(config, "action_aware_depth", 3)),
            dropout=float(getattr(config, "action_aware_dropout", 0.0)),
        ).to(self.device, dtype=torch.float32)

        self.optimizer.add_param_group({
            "params": [p for p in self.action_aware_head.parameters() if p.requires_grad],
            "lr": float(getattr(config, "action_aware_learning_rate", config.learning_rate)),
            "weight_decay": float(
                getattr(config, "action_aware_weight_decay", config.weight_decay)
            ),
        })
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=[
                lambda step: warmup_constant_lambda(step, config.warmup_steps)
                for _ in self.optimizer.param_groups
            ],
        )

        self._load_action_aware_head(getattr(config, "resume_from", None))
        self._broadcast_head()

        self._captured = {}
        self.transformer.action_proj_out.register_forward_pre_hook(
            self._make_hook("action")
        )

    def _make_hook(self, name):
        def hook(module, args):
            self._captured[name] = args[0]
        return hook

    def _head_config(self):
        return {
            "action_dim": self.action_dim,
            "action_tokens": self.action_tokens,
            "action_feature_dim": self.action_feature_dim,
            "mutual_action_video_alignment": True,
            "transition_pool_size": self.transition_pool_size,
            "transition_energy_pool_size": self.transition_energy_pool_size,
            "lambda_action_chunk": self.lambda_action_chunk,
            "lambda_soft_contrastive": self.lambda_soft_contrastive,
            "aux_warmup_steps": self.aux_warmup_steps,
            "aux_max_primary_fraction": self.aux_max_primary_fraction,
            "soft_positive_mix": self.soft_positive_mix,
            "teacher_temperature": self.teacher_temperature,
            "student_temperature": self.student_temperature,
            "align_action_frame_offset": self.action_frame_offset,
        }

    def _load_action_aware_head(self, checkpoint_path):
        if not checkpoint_path:
            return
        head_file = Path(checkpoint_path) / "action_aware_mutual_head" / "model.safetensors"
        if not head_file.exists():
            if self.config.rank == 0:
                logger.warning("Action-aware head not found at %s; using a fresh head", head_file)
            return
        result = self.action_aware_head.load_state_dict(
            load_file(head_file, device="cpu"), strict=False
        )
        if self.config.rank == 0:
            logger.info("Loaded action-aware head from %s", head_file)
            if result.missing_keys:
                logger.warning("Missing action-aware keys: %s", result.missing_keys)
            if result.unexpected_keys:
                logger.warning("Unexpected action-aware keys: %s", result.unexpected_keys)

    def _broadcast_head(self):
        if dist.is_initialized():
            for parameter in self.action_aware_head.parameters():
                dist.broadcast(parameter.data, src=0)

    def _sync_head_grads(self):
        if dist.is_initialized():
            for parameter in self.action_aware_head.parameters():
                if parameter.grad is not None:
                    dist.all_reduce(parameter.grad, op=dist.ReduceOp.AVG)

    @staticmethod
    def _metric_item(value):
        if hasattr(value, "to_local"):
            value = value.to_local()
        if torch.is_tensor(value):
            return value.detach().float().cpu().item()
        return float(value)

    def _latent_prediction_tensor(self, latent_prediction, targets):
        return data_seq_to_patch(
            self.patch_size,
            latent_prediction,
            targets.shape[-3],
            targets.shape[-2],
            targets.shape[-1],
            batch_size=latent_prediction.shape[0],
        ).float()

    def _sigma_for_timesteps(self, scheduler, timesteps):
        flat_timesteps = timesteps.reshape(-1)
        schedule_timesteps = scheduler.timesteps.to(flat_timesteps.device)
        timestep_ids = torch.argmin(
            (schedule_timesteps[:, None] - flat_timesteps[None]).abs(), dim=0
        )
        sigmas = scheduler.sigmas.to(flat_timesteps.device)[timestep_ids]
        return sigmas.reshape(timesteps.shape)

    def _predicted_clean_video(self, input_dict, latent_prediction):
        targets = input_dict["latent_dict"]["targets"]
        prediction = self._latent_prediction_tensor(latent_prediction, targets)
        noisy = input_dict["latent_dict"]["noisy_latents"].float()
        sigma = self._sigma_for_timesteps(
            self.train_scheduler_latent,
            input_dict["latent_dict"]["timesteps"],
        )[:, None, :, None, None]
        # Flow target is epsilon - x0 and x_t=(1-sigma)x0+sigma*epsilon.
        return noisy - sigma * prediction

    def _video_transition_features(self, predicted_clean_video):
        delta = predicted_clean_video[:, :, 1:] - predicted_clean_video[:, :, :-1]
        batch, channels, frames, height, width = delta.shape
        delta = delta.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, channels, height, width
        )
        signed = F.adaptive_avg_pool2d(
            delta, (self.transition_pool_size, self.transition_pool_size)
        ).flatten(1)
        energy = F.adaptive_avg_pool2d(
            delta.abs(),
            (self.transition_energy_pool_size, self.transition_energy_pool_size),
        ).flatten(1)
        return torch.cat([signed, energy], dim=-1), batch, frames

    def _paired_actions(self, input_dict, transition_frames):
        actions = input_dict["action_dict"]["latent"].float()
        masks = input_dict["action_dict"]["actions_mask"].bool()
        # [B, C, F, N, 1] -> [B, F, N, C]
        actions = actions.squeeze(-1).permute(0, 2, 3, 1)
        masks = masks.squeeze(-1).permute(0, 2, 3, 1)

        start = max(self.action_frame_offset, 0)
        end = start + transition_frames
        if end > actions.shape[1]:
            start, end = 0, transition_frames
        actions = actions[:, start:end].reshape(-1, self.action_tokens * self.action_dim)
        masks = masks[:, start:end].reshape(-1, self.action_tokens * self.action_dim)
        return actions.detach(), masks.detach()

    def _paired_action_branch_features(self, input_dict, transition_frames):
        if "action" not in self._captured:
            return None
        action_hidden = self._captured["action"]
        actions = input_dict["action_dict"]["latent"]
        masks = input_dict["action_dict"]["actions_mask"].bool()
        batch = actions.shape[0]
        action_frames = actions.shape[2]
        hidden_dim = action_hidden.shape[-1]
        action_hidden = action_hidden.reshape(batch, action_frames, -1, hidden_dim).float()

        # [B, C, F, N, 1] -> token-valid mask [B, F, N]
        token_mask = masks.any(dim=1).squeeze(-1).bool()
        token_mask_f = token_mask.unsqueeze(-1).float()
        action_features = (
            action_hidden * token_mask_f
        ).sum(dim=2) / token_mask_f.sum(dim=2).clamp_min(1.0)

        start = max(self.action_frame_offset, 0)
        end = start + transition_frames
        if end > action_features.shape[1]:
            start, end = 0, transition_frames
        return action_features[:, start:end].reshape(-1, hidden_dim)

    def _gather_with_grad(self, tensor):
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return tensor, 0
        try:
            from torch.distributed.nn.functional import all_gather
            gathered = all_gather(tensor.contiguous())
            if isinstance(gathered, tuple):
                gathered = list(gathered)
            offset = dist.get_rank() * tensor.shape[0]
            return torch.cat(gathered, dim=0), offset
        except Exception:
            # Keep training robust if differentiable all_gather is unavailable.
            # The local symmetric objective still gives gradients to both branches.
            return tensor, 0

    def _gather_detached(self, tensor):
        tensor = tensor.detach()
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return tensor, 0
        gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor.contiguous())
        offset = dist.get_rank() * tensor.shape[0]
        return torch.cat(gathered, dim=0), offset

    def _mutual_soft_contrastive_loss(
        self,
        video_embedding,
        action_embedding,
        action_target,
        action_mask,
    ):
        target_mask = action_mask.float()
        target = F.normalize(action_target.float() * target_mask, dim=-1, eps=1e-6)
        candidate_target, positive_offset = self._gather_detached(target)

        if candidate_target.shape[0] < 2:
            zero = video_embedding.new_zeros(())
            return zero, zero.detach(), zero.detach()

        video_embedding = F.normalize(video_embedding.float(), dim=-1, eps=1e-6)
        action_embedding = F.normalize(action_embedding.float(), dim=-1, eps=1e-6)
        candidate_video, _ = self._gather_with_grad(video_embedding)
        candidate_action, _ = self._gather_with_grad(action_embedding)
        if (
            candidate_video.shape[0] != candidate_target.shape[0]
            or candidate_action.shape[0] != candidate_target.shape[0]
        ):
            candidate_target = target
            positive_offset = 0

        with torch.no_grad():
            teacher_logits = target @ candidate_target.t() / self.teacher_temperature
            soft_targets = F.softmax(teacher_logits, dim=-1)
            one_hot = torch.zeros_like(soft_targets)
            local_ids = torch.arange(target.shape[0], device=target.device)
            positive_ids = local_ids + positive_offset
            if positive_ids.max().item() < one_hot.shape[1]:
                one_hot[local_ids, positive_ids] = 1.0
            target_distribution = (
                (1.0 - self.soft_positive_mix) * one_hot
                + self.soft_positive_mix * soft_targets
            )

        logits_v2a = video_embedding @ candidate_action.t() / self.student_temperature
        logits_a2v = action_embedding @ candidate_video.t() / self.student_temperature
        loss_v2a = -(
            target_distribution * F.log_softmax(logits_v2a, dim=-1)
        ).sum(-1).mean()
        loss_a2v = -(
            target_distribution * F.log_softmax(logits_a2v, dim=-1)
        ).sum(-1).mean()
        loss = 0.5 * (loss_v2a + loss_a2v)

        with torch.no_grad():
            v_top1 = (logits_v2a.argmax(dim=-1) == positive_ids).float().mean()
            a_top1 = (logits_a2v.argmax(dim=-1) == positive_ids).float().mean()
            top1 = 0.5 * (v_top1 + a_top1)
            soft_positive_mass = (
                target_distribution.sum(dim=-1)
                - target_distribution[local_ids, positive_ids.clamp(max=target_distribution.shape[1] - 1)]
            ).mean()
        return loss, top1, soft_positive_mass

    def _compute_action_aware_losses(self, input_dict, latent_prediction):
        predicted_clean = self._predicted_clean_video(input_dict, latent_prediction)
        video_features, _, transition_frames = self._video_transition_features(predicted_clean)
        action_target, action_mask = self._paired_actions(input_dict, transition_frames)
        action_features = self._paired_action_branch_features(input_dict, transition_frames)

        valid = action_mask.any(dim=-1)
        if action_features is None:
            valid = valid & torch.zeros_like(valid)
        video_features = video_features[valid]
        action_target = action_target[valid]
        action_mask = action_mask[valid]
        if action_features is not None:
            action_features = action_features[valid]
        if video_features.shape[0] == 0:
            zero = latent_prediction.new_zeros(())
            return zero, zero, zero.detach(), zero.detach(), 0

        video_embedding, action_embedding, action_prediction = self.action_aware_head(
            video_features,
            action_features,
        )
        chunk_loss = masked_smooth_l1(
            action_prediction,
            action_target,
            action_mask,
            beta=0.1,
        )
        soft_loss, top1, soft_positive_mass = self._mutual_soft_contrastive_loss(
            video_embedding,
            action_embedding,
            action_target,
            action_mask,
        )
        return chunk_loss, soft_loss, top1, soft_positive_mass, int(valid.sum().item())

    def _auxiliary_weight(self, primary_loss, chunk_loss, soft_loss):
        warmup = min(1.0, float(self.step + 1) / max(1, self.aux_warmup_steps))
        nominal = warmup * (
            self.lambda_action_chunk * chunk_loss
            + self.lambda_soft_contrastive * soft_loss
        ) / self.gradient_accumulation_steps
        if self.aux_max_primary_fraction <= 0:
            cap_scale = nominal.detach().new_tensor(1.0)
        else:
            allowed = self.aux_max_primary_fraction * primary_loss.detach()
            cap_scale = (allowed / nominal.detach().clamp_min(1e-8)).clamp(max=1.0)
        return nominal * cap_scale, warmup, cap_scale

    def _train_step(self, batch, batch_idx):
        batch = self.convert_input_format(batch)
        input_dict = self._prepare_input_dict(batch)
        should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0
        self.transformer.set_requires_gradient_sync(should_sync)

        self._captured = {}
        output = self.transformer(input_dict, train_mode=True)
        latent_loss, action_loss = self.compute_loss(input_dict, output)
        chunk_loss, soft_loss, top1, soft_mass, valid_pairs = (
            self._compute_action_aware_losses(input_dict, output[0])
        )
        primary_loss = latent_loss + action_loss
        auxiliary_loss, warmup, cap_scale = self._auxiliary_weight(
            primary_loss, chunk_loss, soft_loss
        )
        (primary_loss + auxiliary_loss).backward()

        losses = {
            "latent_loss": latent_loss.detach(),
            "action_loss": action_loss.detach(),
            "chunk_loss": chunk_loss.detach(),
            "soft_loss": soft_loss.detach(),
            "top1": top1.detach(),
            "soft_positive_mass": soft_mass.detach(),
            "auxiliary_loss": auxiliary_loss.detach(),
            "aux_warmup": warmup,
            "aux_cap_scale": cap_scale.detach(),
            "valid_pairs": valid_pairs,
        }

        if should_sync:
            self._sync_head_grads()
            losses["total_norm"] = torch.nn.utils.clip_grad_norm_(
                self.transformer.parameters(), 2.0
            )
            losses["head_norm"] = torch.nn.utils.clip_grad_norm_(
                self.action_aware_head.parameters(), 2.0, foreach=False
            )
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            losses["should_log"] = True
        else:
            losses["should_log"] = False
        return losses

    def save_checkpoint(self):
        super().save_checkpoint()
        if self.config.rank == 0:
            checkpoint_dir = self.save_dir / f"checkpoint_step_{self.step}"
            head_dir = checkpoint_dir / "action_aware_mutual_head"
            head_dir.mkdir(parents=True, exist_ok=True)
            state = {
                key: value.detach().float().cpu().contiguous()
                for key, value in self.action_aware_head.state_dict().items()
            }
            save_file(state, head_dir / "model.safetensors")
            with open(head_dir / "config.json", "w") as handle:
                json.dump(self._head_config(), handle, indent=2)
            logger.info("Saved action-aware head to %s", head_dir)
        if dist.is_initialized():
            dist.barrier()

    def train(self):
        logger.info("Starting mutual action-aware alignment training for %s steps", self.config.num_steps)
        self.transformer.train()
        self.action_aware_head.train()
        progress = tqdm(
            total=self.config.num_steps,
            desc="Action-aware training",
            disable=(self.config.rank != 0),
            leave=True,
            dynamic_ncols=True,
            initial=self.step,
        )
        self.optimizer.zero_grad()
        accumulated = {
            key: [] for key in (
                "latent_loss", "action_loss", "chunk_loss", "soft_loss",
                "top1", "soft_positive_mass", "auxiliary_loss",
            )
        }
        step_in_accumulation = 0

        while self.step < self.config.num_steps:
            losses = self._train_step(self._get_next_batch(), step_in_accumulation)
            for key in accumulated:
                accumulated[key].append(losses[key])
            step_in_accumulation += 1

            if losses["should_log"]:
                metrics = {
                    key: dist_mean(torch.stack(values).mean()).detach().cpu().item()
                    for key, values in accumulated.items()
                }
                max_chunk = dist_max(torch.stack(accumulated["chunk_loss"]).mean()).detach().cpu().item()
                accumulated = {key: [] for key in accumulated}
                step_in_accumulation = 0

                torch.cuda.synchronize()
                if self.step % self.config.gc_interval == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                if self.config.rank == 0:
                    total_norm = self._metric_item(losses["total_norm"])
                    head_norm = self._metric_item(losses["head_norm"])
                    cap_scale = self._metric_item(losses["aux_cap_scale"])
                    lr = self.lr_scheduler.get_last_lr()[0]
                    head_lr = self.lr_scheduler.get_last_lr()[-1]
                    progress.n += 1
                    progress.set_postfix({
                        "video": f"{metrics['latent_loss']:.3f}",
                        "action": f"{metrics['action_loss']:.3f}",
                        "chunk": f"{metrics['chunk_loss']:.3f}",
                        "soft": f"{metrics['soft_loss']:.3f}",
                        "aux": f"{metrics['auxiliary_loss']:.3f}",
                        "r1": f"{metrics['top1']:.2f}",
                        "step": self.step,
                    })
                    if self.config.enable_wandb:
                        self.wandb.log({
                            "loss_metrics/global_avg_video_loss": metrics["latent_loss"],
                            "loss_metrics/global_avg_action_loss": metrics["action_loss"],
                            "action_aware/action_chunk_loss": metrics["chunk_loss"],
                            "action_aware/soft_contrastive_loss": metrics["soft_loss"],
                            "action_aware/weighted_auxiliary_loss": metrics["auxiliary_loss"],
                            "action_aware/retrieval_top1": metrics["top1"],
                            "action_aware/soft_positive_mass": metrics["soft_positive_mass"],
                            "action_aware/aux_warmup": losses["aux_warmup"],
                            "action_aware/aux_cap_scale": cap_scale,
                            "action_aware/valid_pairs": losses["valid_pairs"],
                            "action_aware/max_chunk_loss": max_chunk,
                            "grad_norm": total_norm,
                            "action_aware/head_grad_norm": head_norm,
                            "lr": lr,
                            "action_aware/head_lr": head_lr,
                        }, step=self.step)

                self.step += 1
                if self.step % self.config.save_interval == 0:
                    if self.config.rank == 0:
                        logger.info("Starting save model at step %s", self.step)
                    self.save_checkpoint()

            if dist.is_initialized():
                dist.barrier()

        progress.close()
        logger.info("Mutual action-aware alignment training completed")


def run(args):
    config = copy.deepcopy(VA_CONFIGS[args.config_name])
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    init_distributed(world_size, local_rank, rank)
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size

    overrides = {
        "save_root": args.save_root,
        "resume_from": args.resume_from,
        "num_steps": args.num_steps,
        "save_interval": args.save_interval,
        "batch_size": args.batch_size,
        "lambda_action_chunk": args.lambda_action_chunk,
        "lambda_soft_contrastive": args.lambda_soft_contrastive,
        "aux_warmup_steps": args.aux_warmup_steps,
        "aux_max_primary_fraction": args.aux_max_primary_fraction,
        "soft_positive_mix": args.soft_positive_mix,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(config, key, value)
    config.enable_wandb = bool(args.enable_wandb and not args.disable_wandb)

    if rank == 0:
        logger.info("Using config: %s", args.config_name)
        logger.info("World size: %s; per-GPU batch: %s", world_size, config.batch_size)
        logger.info("Output: %s", config.save_root)
        logger.info(
            "Auxiliary weights: chunk=%s soft=%s warmup=%s cap=%s",
            getattr(config, "lambda_action_chunk", 0.1),
            getattr(config, "lambda_soft_contrastive", 0.05),
            getattr(config, "aux_warmup_steps", 250),
            getattr(config, "aux_max_primary_fraction", 0.15),
        )
    ActionAwareAlignTrainer(config).train()


def main():
    parser = argparse.ArgumentParser(description="Train mutual action-aware video/action transitions")
    parser.add_argument("--config-name", default="robotwin_contrastive_align")
    parser.add_argument("--save-root", default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lambda-action-chunk", type=float, default=None)
    parser.add_argument("--lambda-soft-contrastive", type=float, default=None)
    parser.add_argument("--aux-warmup-steps", type=int, default=None)
    parser.add_argument("--aux-max-primary-fraction", type=float, default=None)
    parser.add_argument("--soft-positive-mix", type=float, default=None)
    parser.add_argument("--enable-wandb", action="store_true")
    parser.add_argument("--disable-wandb", action="store_true")
    args = parser.parse_args()
    if args.enable_wandb and args.disable_wandb:
        raise ValueError("--enable-wandb and --disable-wandb are mutually exclusive")
    run(args)


if __name__ == "__main__":
    init_logger()
    main()
