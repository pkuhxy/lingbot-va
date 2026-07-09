from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


TOKEN_GROUPS = (
    "noisy_video", "condition_video", "noisy_action", "condition_action"
)


def automatic_layers(num_layers: int) -> list[int]:
    if num_layers < 1:
        raise ValueError("Transformer must contain at least one block")
    return sorted({0, round((num_layers - 1) / 3), round(2 * (num_layers - 1) / 3), num_layers - 1})


@dataclass(frozen=True)
class TokenLayout:
    batch_size: int
    video_grid: tuple[int, int, int]
    action_grid: tuple[int, int, int]

    @property
    def video_tokens(self) -> int:
        f, h, w = self.video_grid
        return f * h * w

    @property
    def action_tokens(self) -> int:
        f, h, w = self.action_grid
        return f * h * w

    @property
    def total_unpadded(self) -> int:
        return 2 * self.batch_size * (self.video_tokens + self.action_tokens)

    def slices(self) -> dict[str, slice]:
        bv, ba = self.batch_size * self.video_tokens, self.batch_size * self.action_tokens
        return {
            "noisy_video": slice(0, bv),
            "condition_video": slice(bv, 2 * bv),
            "noisy_action": slice(2 * bv, 2 * bv + ba),
            "condition_action": slice(2 * bv + ba, 2 * bv + 2 * ba),
        }


def token_layout(input_dict: dict, patch_size=(1, 2, 2)) -> TokenLayout:
    latent = input_dict["latent_dict"]["noisy_latents"]
    action = input_dict["action_dict"]["noisy_latents"]
    b, _, fv, hv, wv = latent.shape
    ba, _, fa, ha, wa = action.shape
    if b != ba:
        raise ValueError("Video/action batch sizes differ")
    pf, ph, pw = patch_size
    return TokenLayout(b, (fv // pf, hv // ph, wv // pw), (fa, ha, wa))


def split_tokens(output: torch.Tensor, layout: TokenLayout) -> dict[str, torch.Tensor]:
    if output.ndim != 3 or output.shape[0] != 1:
        raise ValueError(f"Expected hook output [1,L,C], got {tuple(output.shape)}")
    if output.shape[1] < layout.total_unpadded:
        raise ValueError("Hook output is shorter than the four non-padding groups")
    result = {}
    for name, group_slice in layout.slices().items():
        tokens = output[:, group_slice, :].reshape(layout.batch_size, -1, output.shape[-1])
        result[name] = tokens
    return result


def pool_token_group(tokens: torch.Tensor, grid: tuple[int, int, int]) -> dict[str, torch.Tensor]:
    b, length, channels = tokens.shape
    frames, height, width = grid
    if length != frames * height * width:
        raise ValueError(f"Token length {length} does not match grid {grid}")
    spatial = tokens.float().reshape(b, frames, height, width, channels).mean(dim=(2, 3))
    return {"frame": spatial.cpu(), "window": spatial.mean(dim=1).cpu()}


class BlockFeatureHooks:
    def __init__(self, transformer, layers: Iterable[int], layout: TokenLayout):
        self.transformer = transformer
        self.layers = sorted(set(int(layer) for layer in layers))
        self.layout = layout
        self.outputs: dict[int, torch.Tensor] = {}
        self.handles = []

    def __enter__(self):
        for layer in self.layers:
            if not 0 <= layer < len(self.transformer.blocks):
                raise IndexError(f"Hook layer {layer} outside [0, {len(self.transformer.blocks)})")

            def hook(_module, _inputs, output, layer=layer):
                if not torch.is_tensor(output):
                    raise TypeError(f"Block {layer} returned {type(output)}, expected Tensor")
                self.outputs[layer] = output.detach()

            self.handles.append(self.transformer.blocks[layer].register_forward_hook(hook))
        return self

    def pooled(self) -> dict[int, dict[str, dict[str, torch.Tensor]]]:
        missing = set(self.layers).difference(self.outputs)
        if missing:
            raise RuntimeError(f"Hooks did not fire for layers {sorted(missing)}")
        result = {}
        for layer, output in self.outputs.items():
            groups = split_tokens(output, self.layout)
            result[layer] = {
                name: pool_token_group(
                    tokens,
                    self.layout.video_grid if "video" in name else self.layout.action_grid,
                )
                for name, tokens in groups.items()
            }
        return result

    def __exit__(self, *_exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.outputs.clear()

