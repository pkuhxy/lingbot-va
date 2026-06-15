# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS
from dataset import MultiLatentLeRobotDataset
from distributed.util import init_distributed
from modules.latent_posttraining import (
    WAMControllableSubspaceProbe,
    counterfactual_loss,
    masked_mean,
    masked_smooth_l1,
    relational_loss,
    retrieval_metrics,
)
from modules.utils import load_transformer
from utils import get_mesh_id, init_logger, logger, warmup_constant_lambda


def _dist_mean(value):
    if not dist.is_available() or not dist.is_initialized():
        return value
    value = value.detach()
    dist.all_reduce(value, op=dist.ReduceOp.AVG)
    return value


def _cfg_to_dict(config):
    out = {}
    for key, value in config.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        elif isinstance(value, (list, tuple)):
            out[key] = list(value)
        elif isinstance(value, dict):
            out[key] = value
        else:
            out[key] = repr(value)
    return out


class Stage1LatentPostTrainer:
    def __init__(self, config):
        self.config = config
        self.rank = getattr(config, "rank", 0)
        self.local_rank = getattr(config, "local_rank", 0)
        self.world_size = getattr(config, "world_size", 1)
        self.device = torch.device(
            f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
        )
        self.hidden_source = getattr(config, "latent_hidden_source", "embed")
        self.video_delta_mode = getattr(config, "video_delta_mode", "last_first")
        self.gradient_accumulation_steps = getattr(config, "gradient_accumulation_steps", 1)
        self.step = 0
        self.negative_queue = None

        if getattr(config, "enable_wandb", False) and self.rank == 0:
            if wandb is None:
                logger.warning("WandB requested but wandb is not installed; disabling logging")
                self.wandb = None
            else:
                wandb.login(
                    host=os.environ.get("WANDB_BASE_URL"),
                    key=os.environ.get("WANDB_API_KEY"),
                )
                wandb.init(
                    entity=os.environ.get("WANDB_TEAM_NAME"),
                    project=os.getenv("WANDB_PROJECT", "wam_latent_posttrain"),
                    config=_cfg_to_dict(config),
                    name=getattr(config, "wandb_name", "wam_stage1_adapter"),
                )
                self.wandb = wandb
        else:
            self.wandb = None

        self._load_frozen_transformer()
        self._build_probe()
        self._build_data()

        self.save_dir = Path(config.save_root) / "latent_posttrain_stage1"
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _load_frozen_transformer(self):
        if getattr(self.config, "resume_from", None):
            transformer_path = Path(self.config.resume_from) / "transformer"
        else:
            transformer_path = Path(self.config.wan22_pretrained_model_name_or_path) / "transformer"

        logger.info(f"Loading frozen transformer from {transformer_path}")
        self.transformer = load_transformer(
            str(transformer_path),
            torch_dtype=torch.float32,
            torch_device="cpu",
            attn_mode=getattr(self.config, "latent_hidden_attn_mode", "torch"),
        ).to(self.device)
        self.transformer.eval()
        self.transformer.requires_grad_(False)

    def _build_probe(self):
        hidden_dim = self.transformer.num_attention_heads * self.transformer.attention_head_dim
        action_dim = getattr(self.config, "action_dim", self.transformer.config.action_dim)
        probe = WAMControllableSubspaceProbe(
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            subspace_dim=getattr(self.config, "subspace_dim", 256),
            adapter_hidden_dim=getattr(self.config, "adapter_hidden_dim", 1024),
            adapter_depth=getattr(self.config, "adapter_depth", 2),
            dropout=getattr(self.config, "adapter_dropout", 0.0),
        ).to(self.device)

        if self.world_size > 1:
            probe = DistributedDataParallel(
                probe,
                device_ids=[self.local_rank] if self.device.type == "cuda" else None,
            )
        self.probe = probe
        self.optimizer = torch.optim.AdamW(
            self.probe.parameters(),
            lr=getattr(self.config, "latent_learning_rate", self.config.learning_rate),
            betas=(self.config.beta1, self.config.beta2),
            eps=1e-8,
            weight_decay=getattr(self.config, "latent_weight_decay", self.config.weight_decay),
        )
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: warmup_constant_lambda(step, self.config.warmup_steps),
        )

    def _build_data(self):
        dataset = MultiLatentLeRobotDataset(
            config=self.config,
            num_init_worker=getattr(self.config, "dataset_init_worker", 8),
        )
        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                seed=getattr(self.config, "seed", 42),
            )
            if self.world_size > 1
            else None
        )
        self.train_loader = DataLoader(
            dataset,
            batch_size=getattr(self.config, "latent_batch_size", self.config.batch_size),
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.config.load_worker,
            pin_memory=True,
            drop_last=True,
        )
        self.train_loader_iter = None

    def _get_next_batch(self):
        if self.train_loader_iter is None:
            self.train_loader_iter = iter(self.train_loader)
        try:
            return next(self.train_loader_iter)
        except StopIteration:
            if hasattr(self.train_loader.sampler, "set_epoch"):
                epoch = self.step // max(len(self.train_loader), 1)
                self.train_loader.sampler.set_epoch(epoch)
            self.train_loader_iter = iter(self.train_loader)
            return next(self.train_loader_iter)

    def _move_batch(self, batch):
        return {
            key: value.to(self.device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

    def _grid_id(self, latent, action_mode):
        patch_f, patch_h, patch_w = self.transformer.patch_size
        if action_mode:
            patch_f = patch_h = patch_w = 1
        grid_id = get_mesh_id(
            latent.shape[-3] // patch_f,
            latent.shape[-2] // patch_h,
            latent.shape[-1] // patch_w,
            t=1 if action_mode else 0,
            f_w=1,
            f_shift=0,
            action=action_mode,
        ).to(self.device)
        return grid_id[None].repeat(latent.shape[0], 1, 1)

    def _clean_transformer_input(self, batch):
        latents = batch["latents"]
        actions = batch["actions"]
        latent_steps = torch.zeros(
            (latents.shape[0], latents.shape[-3]),
            dtype=torch.float32,
            device=self.device,
        )
        action_steps = torch.zeros(
            (actions.shape[0], actions.shape[-3]),
            dtype=torch.float32,
            device=self.device,
        )
        return {
            "latent_dict": {
                "noisy_latents": latents,
                "latent": latents,
                "timesteps": latent_steps,
                "cond_timesteps": latent_steps,
                "grid_id": self._grid_id(latents, action_mode=False),
                "text_emb": batch["text_emb"],
            },
            "action_dict": {
                "noisy_latents": actions,
                "latent": actions,
                "timesteps": action_steps,
                "cond_timesteps": action_steps,
                "grid_id": self._grid_id(actions, action_mode=True),
                "text_emb": batch["text_emb"],
                "actions_mask": batch["actions_mask"],
            },
            "chunk_size": getattr(self.config, "stage1_chunk_size", self.config.frame_chunk_size),
            "window_size": getattr(self.config, "stage1_window_size", self.config.attn_window),
            "return_hidden": True,
        }

    @torch.no_grad()
    def _extract_branch_hiddens(self, batch):
        latents = batch["latents"].float()
        actions = batch["actions"].float()
        if self.hidden_source == "transformer":
            hidden = self.transformer(self._clean_transformer_input(batch), train_mode=True)
            latent_tokens = hidden["latent_hidden"].float()
            action_tokens = hidden["action_hidden"].float()
        elif self.hidden_source == "embed":
            latent_tokens = self.transformer._input_embed(latents, input_type="latent").float()
            action_tokens = self.transformer._input_embed(actions, input_type="action").float()
        else:
            raise ValueError(f"Unsupported latent_hidden_source: {self.hidden_source}")

        video_delta_tokens = self._video_delta_tokens(latent_tokens, latents)
        action_token_mask = batch["actions_mask"].any(dim=1).flatten(1).bool()
        action_targets = actions.permute(0, 2, 3, 4, 1).reshape(actions.shape[0], -1, actions.shape[1])
        action_mean = masked_mean(action_targets, action_token_mask, dim=1)
        return {
            "video_delta_tokens": video_delta_tokens,
            "action_tokens": action_tokens,
            "action_token_mask": action_token_mask,
            "action_targets": action_targets,
            "action_mean": action_mean,
        }

    def _video_delta_tokens(self, latent_tokens, latents):
        patch_f, patch_h, patch_w = self.transformer.patch_size
        batch_size, _, frames, height, width = latents.shape
        frame_tokens = frames // patch_f
        spatial_tokens = (height // patch_h) * (width // patch_w)
        latent_tokens = latent_tokens.reshape(batch_size, frame_tokens, spatial_tokens, -1)

        if frame_tokens < 2:
            return latent_tokens[:, :1].new_zeros(batch_size, spatial_tokens, latent_tokens.shape[-1])
        if self.video_delta_mode == "adjacent":
            return (latent_tokens[:, 1:] - latent_tokens[:, :-1]).reshape(
                batch_size, (frame_tokens - 1) * spatial_tokens, -1
            )
        if self.video_delta_mode == "last_first":
            return latent_tokens[:, -1] - latent_tokens[:, 0]
        raise ValueError(f"Unsupported video_delta_mode: {self.video_delta_mode}")

    def _negative_bank(self):
        if self.negative_queue is None or self.negative_queue.numel() == 0:
            return None
        return self.negative_queue.to(self.device)

    @torch.no_grad()
    def _update_negative_queue(self, c_a):
        queue_size = getattr(self.config, "counterfactual_queue_size", 1024)
        if queue_size <= 0:
            return
        c_a = c_a.detach().float()
        if self.negative_queue is None:
            self.negative_queue = c_a
        else:
            self.negative_queue = torch.cat([self.negative_queue, c_a], dim=0)
        if self.negative_queue.shape[0] > queue_size:
            self.negative_queue = self.negative_queue[-queue_size:]

    def _compute_loss(self, features):
        output = self.probe(
            features["video_delta_tokens"],
            features["action_tokens"],
            features["action_token_mask"],
        )
        recon = masked_smooth_l1(
            output["action_recon"],
            features["action_targets"],
            features["action_token_mask"],
            beta=getattr(self.config, "huber_beta", 0.1),
        )
        inverse = masked_smooth_l1(
            output["action_from_video"],
            features["action_mean"],
            beta=getattr(self.config, "huber_beta", 0.1),
        )
        cf = counterfactual_loss(
            output["c_v"],
            output["c_a"],
            negative_c_a=self._negative_bank(),
            tau=getattr(self.config, "counterfactual_tau", 0.1),
            margin=getattr(self.config, "counterfactual_margin", None),
        )
        rel = relational_loss(output["c_v"], output["c_a"])

        total = (
            getattr(self.config, "lambda_recon_action", 1.0) * recon
            + getattr(self.config, "lambda_inverse", 1.0) * inverse
            + getattr(self.config, "lambda_counterfactual", 0.2) * cf
            + getattr(self.config, "lambda_relational", 0.0) * rel
        )
        metrics = {
            "loss": total.detach(),
            "loss_recon_action": recon.detach(),
            "loss_inverse": inverse.detach(),
            "loss_counterfactual": cf.detach(),
            "loss_relational": rel.detach(),
        }
        metrics.update({k: v.detach() for k, v in retrieval_metrics(output["c_v"], output["c_a"]).items()})
        return total, output, metrics

    def _save_checkpoint(self):
        if self.rank != 0:
            return
        checkpoint_dir = self.save_dir / f"checkpoint_step_{self.step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        probe = self.probe.module if hasattr(self.probe, "module") else self.probe
        torch.save(
            {
                "step": self.step,
                "probe": probe.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": _cfg_to_dict(self.config),
            },
            checkpoint_dir / "stage1_probe.pt",
        )
        with open(checkpoint_dir / "config.json", "w") as f:
            json.dump(_cfg_to_dict(self.config), f, indent=2)
        logger.info(f"Saved Stage-1 probe checkpoint to {checkpoint_dir}")

    def train(self):
        logger.info(
            f"Starting WAM latent Stage-1 adapter training for {self.config.num_steps} steps"
        )
        progress = tqdm(
            total=self.config.num_steps,
            desc="Stage1",
            disable=(self.rank != 0),
            dynamic_ncols=True,
            initial=self.step,
        )
        self.optimizer.zero_grad(set_to_none=True)
        accum_metrics = {}
        micro_step = 0

        while self.step < self.config.num_steps:
            batch = self._move_batch(self._get_next_batch())
            features = self._extract_branch_hiddens(batch)
            loss, output, metrics = self._compute_loss(features)
            (loss / self.gradient_accumulation_steps).backward()
            self._update_negative_queue(output["c_a"])

            for key, value in metrics.items():
                accum_metrics.setdefault(key, []).append(value)
            micro_step += 1

            if micro_step % self.gradient_accumulation_steps != 0:
                continue

            total_norm = torch.nn.utils.clip_grad_norm_(
                self.probe.parameters(),
                getattr(self.config, "max_grad_norm", 2.0),
            )
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

            log_metrics = {
                key: _dist_mean(torch.stack(values).mean()).detach().cpu().item()
                for key, values in accum_metrics.items()
            }
            accum_metrics = {}
            micro_step = 0

            if self.rank == 0:
                lr = self.lr_scheduler.get_last_lr()[0]
                progress.n += 1
                postfix = {
                    "loss": f"{log_metrics['loss']:.4f}",
                    "inv": f"{log_metrics['loss_inverse']:.4f}",
                    "cf": f"{log_metrics['loss_counterfactual']:.4f}",
                    "lr": f"{lr:.2e}",
                }
                if "retrieval_r1" in log_metrics:
                    postfix["r1"] = f"{log_metrics['retrieval_r1']:.3f}"
                progress.set_postfix(postfix)
                if self.wandb is not None:
                    self.wandb.log(
                        {
                            **{f"stage1/{k}": v for k, v in log_metrics.items()},
                            "stage1/grad_norm": total_norm.item(),
                            "stage1/lr": lr,
                        },
                        step=self.step,
                    )

            self.step += 1
            if self.step % self.config.save_interval == 0:
                self._save_checkpoint()
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

        progress.close()
        self._save_checkpoint()
        logger.info("Stage-1 latent adapter training completed")


def run(args):
    config = VA_CONFIGS[args.config_name]
    if args.save_root is not None:
        config.save_root = args.save_root
    if args.dataset_path is not None:
        config.dataset_path = args.dataset_path
        config.empty_emb_path = os.path.join(args.dataset_path, "empty_emb.pt")
    if args.pretrained_model is not None:
        config.wan22_pretrained_model_name_or_path = args.pretrained_model
    if args.hidden_source is not None:
        config.latent_hidden_source = args.hidden_source
    if args.attn_mode is not None:
        config.latent_hidden_attn_mode = args.attn_mode
    if args.num_steps is not None:
        config.num_steps = args.num_steps

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        init_distributed(world_size, local_rank, rank)
    elif torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size

    if rank == 0:
        logger.info(f"Using config: {args.config_name}")
        logger.info(f"World size: {world_size}, Local rank: {local_rank}")
        logger.info(f"Latent hidden source: {config.latent_hidden_source}")

    trainer = Stage1LatentPostTrainer(config)
    trainer.train()


def main():
    parser = argparse.ArgumentParser(
        description="Stage-1 asymmetric controllable-subspace post-training for WAM"
    )
    parser.add_argument("--config-name", default="robotwin_latent_posttrain")
    parser.add_argument("--save-root", default=None)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--pretrained-model", default=None)
    parser.add_argument("--hidden-source", choices=["embed", "transformer"], default=None)
    parser.add_argument("--attn-mode", choices=["torch", "flashattn", "flex"], default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    init_logger()
    main()
