"""CPU-only smoke tests for stage-0 deterministic and metric utilities."""

import numpy as np
import torch

from common.deterministic import deterministic_noise, tensor_sha256
from common.feature_hooks import TokenLayout, automatic_layers, pool_token_group, split_tokens
from common.metrics import cosine_drift, linear_cka
from common.perturbations import apply_perturbation, directional_pair


def main():
    reference = torch.zeros(2, 3, 4)
    first = deterministic_noise(reference, 20260629, "sample", "video")
    second = deterministic_noise(reference, 20260629, "sample", "video")
    assert tensor_sha256(first) == tensor_sha256(second)
    assert automatic_layers(30) == [0, 10, 19, 29]

    layout = TokenLayout(1, (2, 2, 2), (2, 3, 1))
    output = torch.arange((layout.total_unpadded + 17) * 4).reshape(1, -1, 4)
    groups = split_tokens(output, layout)
    assert groups["condition_video"].shape == (1, 8, 4)
    assert pool_token_group(groups["condition_video"], layout.video_grid)["window"].shape == (1, 4)

    features = torch.randn(32, 12, generator=torch.Generator().manual_seed(7))
    assert linear_cka(features, features) > 0.999999
    assert cosine_drift(features, features)["mean"] < 1e-10

    clip = np.full((3, 8, 8, 3), 100, dtype=np.uint8)
    assert np.array_equal(apply_perturbation(clip, "identity", 0, 1), clip)
    plus, minus = directional_pair(clip, "photo", 0.5, 1)
    assert plus.shape == minus.shape == clip.shape
    print("stage0 self-test: PASS")


if __name__ == "__main__":
    main()
