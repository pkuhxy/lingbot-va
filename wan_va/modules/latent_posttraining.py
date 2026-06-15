# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(x, mask=None, dim=1, eps=1e-6):
    if mask is None:
        return x.mean(dim=dim)
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.to(dtype=x.dtype, device=x.device)
    return (x * mask).sum(dim=dim) / mask.sum(dim=dim).clamp_min(eps)


def masked_smooth_l1(pred, target, mask=None, beta=0.1, eps=1e-6):
    loss = F.smooth_l1_loss(pred.float(), target.float(), beta=beta, reduction="none")
    if mask is None:
        return loss.mean()
    while mask.ndim < loss.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.to(dtype=loss.dtype, device=loss.device)
    denom = mask.expand_as(loss).sum().clamp_min(eps)
    return (loss * mask).sum() / denom


class MLP(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim=None,
        depth=2,
        dropout=0.0,
        use_layernorm=True,
    ):
        super().__init__()
        hidden_dim = hidden_dim or max(in_dim, out_dim)
        layers = []
        cur_dim = in_dim
        for _ in range(max(depth - 1, 0)):
            layers.append(nn.Linear(cur_dim, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            cur_dim = hidden_dim
        layers.append(nn.Linear(cur_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class WAMControllableSubspaceProbe(nn.Module):
    """Stage-1 adapter/probe for asymmetric controllable-subspace training.

    The module is intentionally small. The frozen WAM produces video and action
    branch hiddens; this probe learns c_v and c_a plus lightweight decoders.
    """

    def __init__(
        self,
        hidden_dim,
        action_dim,
        subspace_dim=256,
        adapter_hidden_dim=1024,
        adapter_depth=2,
        dropout=0.0,
    ):
        super().__init__()
        self.video_adapter = MLP(
            hidden_dim,
            subspace_dim,
            hidden_dim=adapter_hidden_dim,
            depth=adapter_depth,
            dropout=dropout,
        )
        self.action_adapter = MLP(
            hidden_dim,
            subspace_dim,
            hidden_dim=adapter_hidden_dim,
            depth=adapter_depth,
            dropout=dropout,
        )
        self.action_decoder = MLP(
            subspace_dim,
            action_dim,
            hidden_dim=adapter_hidden_dim,
            depth=adapter_depth,
            dropout=dropout,
        )
        self.inverse_decoder = MLP(
            subspace_dim,
            action_dim,
            hidden_dim=adapter_hidden_dim,
            depth=adapter_depth,
            dropout=dropout,
        )

    def forward(self, video_delta_tokens, action_tokens, action_token_mask=None):
        c_v_tokens = self.video_adapter(video_delta_tokens)
        c_a_tokens = self.action_adapter(action_tokens)

        c_v = c_v_tokens.mean(dim=1)
        c_a = masked_mean(c_a_tokens, action_token_mask, dim=1)

        action_recon = self.action_decoder(c_a_tokens)
        action_from_video = self.inverse_decoder(c_v)
        return {
            "c_v_tokens": c_v_tokens,
            "c_a_tokens": c_a_tokens,
            "c_v": c_v,
            "c_a": c_a,
            "action_recon": action_recon,
            "action_from_video": action_from_video,
        }


def counterfactual_loss(c_v, c_a, negative_c_a=None, tau=0.1, margin=None):
    if c_v.shape[0] < 2 and negative_c_a is None:
        return c_v.new_zeros(())

    c_v = F.normalize(c_v.float(), dim=-1)
    c_a = F.normalize(c_a.float(), dim=-1)
    pos = (c_v * c_a).sum(dim=-1)

    if negative_c_a is None:
        negative_c_a = torch.roll(c_a, shifts=1, dims=0)
    else:
        negative_c_a = F.normalize(negative_c_a.float(), dim=-1)
        if negative_c_a.shape[0] != c_v.shape[0]:
            ids = torch.randint(0, negative_c_a.shape[0], (c_v.shape[0],), device=c_v.device)
            negative_c_a = negative_c_a[ids]

    neg = (c_v * negative_c_a).sum(dim=-1)
    if margin is None:
        return F.softplus(-(pos - neg) / tau).mean()
    return F.relu(margin - pos + neg).mean()


def relational_loss(c_v, c_a):
    if c_v.shape[0] < 2:
        return c_v.new_zeros(())
    c_v = F.normalize(c_v.float(), dim=-1)
    c_a = F.normalize(c_a.float(), dim=-1)
    sim_v = c_v @ c_v.T
    sim_a = c_a @ c_a.T
    sim_v = (sim_v - sim_v.mean(dim=-1, keepdim=True)) / sim_v.std(dim=-1, keepdim=True).clamp_min(1e-6)
    sim_a = (sim_a - sim_a.mean(dim=-1, keepdim=True)) / sim_a.std(dim=-1, keepdim=True).clamp_min(1e-6)
    return F.mse_loss(sim_v, sim_a)


def stage2_latent_consistency_loss(
    probe,
    video_delta_tokens,
    action_tokens,
    action_token_mask=None,
    detach_video_target=True,
):
    """Latent consistency term for Stage-2 WAM action-path post-training."""
    c_v_tokens = probe.video_adapter(video_delta_tokens)
    c_a_tokens = probe.action_adapter(action_tokens)
    c_v = c_v_tokens.mean(dim=1)
    c_a = masked_mean(c_a_tokens, action_token_mask, dim=1)
    if detach_video_target:
        c_v = c_v.detach()
    return F.mse_loss(
        F.normalize(c_a.float(), dim=-1),
        F.normalize(c_v.float(), dim=-1),
    )


@torch.no_grad()
def retrieval_metrics(c_v, c_a):
    if c_v.shape[0] < 2:
        return {}
    c_v = F.normalize(c_v.float(), dim=-1)
    c_a = F.normalize(c_a.float(), dim=-1)
    logits = c_v @ c_a.T
    labels = torch.arange(logits.shape[0], device=logits.device)
    top1 = (logits.argmax(dim=-1) == labels).float().mean()
    ranks = torch.argsort(logits, dim=-1, descending=True)
    reciprocal_rank = 1.0 / ((ranks == labels[:, None]).nonzero()[:, 1].float() + 1.0)
    return {
        "retrieval_r1": top1,
        "retrieval_mrr": reciprocal_rank.mean(),
        "pos_neg_margin": (
            logits.diag()
            - logits.masked_fill(torch.eye(logits.shape[0], device=logits.device).bool(), -1e9).max(dim=-1).values
        ).mean(),
    }
