from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def _matrix(value) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float64)
    if tensor.ndim != 2:
        raise ValueError(f"Expected a matrix, got shape {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise ValueError("Metric input contains NaN/Inf")
    return tensor


def linear_cka(x, y, epsilon=1e-12) -> float:
    x, y = _matrix(x), _matrix(y)
    if x.shape[0] != y.shape[0] or x.shape[0] < 2:
        raise ValueError("CKA inputs must have the same N >= 2")
    x, y = x - x.mean(0), y - y.mean(0)
    numerator = torch.linalg.matrix_norm(x.T @ y).square()
    denominator = torch.linalg.matrix_norm(x.T @ x) * torch.linalg.matrix_norm(y.T @ y)
    return float((numerator / denominator.clamp_min(epsilon)).item())


def cosine_drift(x, y) -> dict[str, float]:
    x, y = _matrix(x), _matrix(y)
    if x.shape != y.shape:
        raise ValueError(f"Cosine drift shapes differ: {x.shape} vs {y.shape}")
    values = 1.0 - F.cosine_similarity(x, y, dim=1, eps=1e-12)
    return {"mean": float(values.mean()), "median": float(values.median())}


def relative_l2(reference: torch.Tensor, value: torch.Tensor, mask=None, epsilon=1e-12) -> float:
    reference, value = reference.float(), value.float()
    if mask is not None:
        mask = torch.broadcast_to(mask.bool(), reference.shape)
        reference, value = reference[mask], value[mask]
    return float(((value - reference).norm() / reference.norm().clamp_min(epsilon)).item())


def normalized_l1(reference: torch.Tensor, value: torch.Tensor, scale, mask=None) -> float:
    delta = (value.float() - reference.float()).abs() / torch.as_tensor(scale).float().clamp_min(1e-12)
    if mask is not None:
        delta = delta[torch.broadcast_to(mask.bool(), delta.shape)]
    return float(delta.mean().item())


def cluster_bootstrap(
    values: Sequence[float], tasks: Sequence[str], clusters: Sequence[str],
    repeats: int = 2000, confidence: float = 0.95, seed: int = 0,
) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    tasks, clusters = np.asarray(tasks), np.asarray(clusters)
    if not (len(values) == len(tasks) == len(clusters)) or len(values) == 0:
        raise ValueError("Bootstrap inputs must have equal non-zero length")
    rng = np.random.default_rng(seed)
    task_names = np.unique(tasks)

    def estimate(index: np.ndarray) -> float:
        return float(np.mean([values[index][tasks[index] == task].mean() for task in task_names if np.any(tasks[index] == task)]))

    estimates = []
    for _ in range(int(repeats)):
        selected = []
        for task in task_names:
            task_idx = np.flatnonzero(tasks == task)
            task_clusters = np.unique(clusters[task_idx])
            sampled = rng.choice(task_clusters, size=len(task_clusters), replace=True)
            selected.extend(np.concatenate([task_idx[clusters[task_idx] == cluster] for cluster in sampled]))
        estimates.append(estimate(np.asarray(selected, dtype=np.int64)))
    alpha = (1.0 - confidence) / 2.0
    return {
        "estimate": estimate(np.arange(len(values))),
        "ci_low": float(np.quantile(estimates, alpha)),
        "ci_high": float(np.quantile(estimates, 1.0 - alpha)),
    }


def l2_normalize(x) -> torch.Tensor:
    return F.normalize(_matrix(x), dim=1, eps=1e-12)


def knn_domain_distance(clean, shifted, k: int) -> float:
    clean, shifted = l2_normalize(clean), l2_normalize(shifted)
    if clean.shape[0] < k:
        raise ValueError(f"Clean bank has {clean.shape[0]} samples, smaller than k={k}")
    distances = torch.cdist(shifted, clean)
    return float(distances.topk(k, largest=False, dim=1).values.mean().item())


def median_bandwidth(clean) -> float:
    clean = _matrix(clean)
    if clean.shape[0] < 2:
        raise ValueError("Bandwidth estimation needs at least two clean samples")
    distances = torch.pdist(clean)
    positive = distances[distances > 0]
    return float(positive.median().item()) if positive.numel() else 1.0


def rbf_mmd2(x, y, bandwidth: float) -> float:
    x, y = _matrix(x), _matrix(y)
    gamma = 1.0 / (2.0 * max(float(bandwidth), 1e-12) ** 2)
    kernel = lambda a, b: torch.exp(-gamma * torch.cdist(a, b).square())
    kxx, kyy, kxy = kernel(x, x), kernel(y, y), kernel(x, y)
    if len(x) > 1:
        xx = (kxx.sum() - kxx.diagonal().sum()) / (len(x) * (len(x) - 1))
    else:
        xx = kxx.mean()
    if len(y) > 1:
        yy = (kyy.sum() - kyy.diagonal().sum()) / (len(y) * (len(y) - 1))
    else:
        yy = kyy.mean()
    return float((xx + yy - 2 * kxy.mean()).item())


def auroc(labels, scores) -> float:
    labels, scores = np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float64)
    positives, negatives = labels.sum(), len(labels) - labels.sum()
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    for score in np.unique(scores):
        tied = scores == score
        ranks[tied] = ranks[tied].mean()
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def domain_logistic_metrics(features, labels, groups, seed=0, c=1.0) -> dict[str, float]:
    x = torch.as_tensor(features, dtype=torch.float64)
    y = torch.as_tensor(labels, dtype=torch.float64)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("Domain classifier needs at least two trajectory groups")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    test_groups = set(unique[: max(1, round(len(unique) * 0.2))])
    test = np.asarray([group in test_groups for group in groups])
    if test.all() or (~test).all():
        raise ValueError("Unable to create group-disjoint classifier split")
    train_x, test_x = x[~test], x[test]
    mean, std = train_x.mean(0), train_x.std(0).clamp_min(1e-8)
    train_x, test_x = (train_x - mean) / std, (test_x - mean) / std
    # Reduce only for numerical tractability; the projection is fitted on train groups.
    rank = min(128, train_x.shape[0] - 1, train_x.shape[1])
    if rank < train_x.shape[1]:
        _, _, basis = torch.pca_lowrank(train_x, q=max(1, rank), center=False)
        train_x, test_x = train_x @ basis, test_x @ basis
    weight = torch.zeros(train_x.shape[1], dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([weight, bias], max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(train_x @ weight + bias, y[~test])
        loss = loss + 0.5 / float(c) * weight.square().sum() / len(train_x)
        loss.backward()
        return loss

    optimizer.step(closure)
    scores = torch.sigmoid(test_x @ weight + bias).detach().numpy()
    truth = y[test].numpy().astype(np.int64)
    prediction = scores >= 0.5
    recalls = [prediction[truth == label].mean() for label in (0, 1) if np.any(truth == label)]
    return {"balanced_accuracy": float(np.mean(recalls)), "auroc": auroc(truth, scores)}


def whiten_residual(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (value.float() - mean.float()) / std.float().clamp_min(1e-6)


def latent_residual_cosine(clean, augmented, clean_current, augmented_current, mean, std) -> float:
    clean_delta = whiten_residual(clean - clean_current, mean, std).flatten()
    aug_delta = whiten_residual(augmented - augmented_current, mean, std).flatten()
    return float((1.0 - F.cosine_similarity(clean_delta[None], aug_delta[None], eps=1e-12)).item())

