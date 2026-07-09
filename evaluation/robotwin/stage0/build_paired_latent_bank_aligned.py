"""Build the paired bank using the exact RGB frame_ids saved by training preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import build_paired_latent_bank as implementation


def aligned_video_window(path: str, start: int, end: int) -> np.ndarray:
    """Decode only the frames used to create the existing clean latent."""
    video_path = Path(path).expanduser().resolve()
    camera = video_path.parent.name
    chunk = video_path.parent.parent.name
    repo = video_path.parents[3]
    latent_path = (
        repo / "latents" / chunk / camera
        / f"{video_path.stem}_{start}_{end}.pth"
    )
    if not latent_path.is_file():
        raise FileNotFoundError(
            f"Training latent metadata not found for RGB alignment: {latent_path}"
        )
    metadata = torch.load(latent_path, map_location="cpu", weights_only=False)
    frame_ids = [int(value) for value in metadata["frame_ids"]]
    wanted = set(frame_ids)
    frames = {}
    for index, frame in enumerate(implementation.iio.imiter(path, plugin="pyav")):
        if index > frame_ids[-1]:
            break
        if index in wanted:
            frames[index] = np.asarray(frame)[..., :3]
    missing = [index for index in frame_ids if index not in frames]
    if missing:
        raise ValueError(
            f"Video {path} is missing aligned frame_ids: {missing[:10]}"
        )
    clip = np.stack([frames[index] for index in frame_ids])
    expected = int(metadata.get("video_num_frames", len(frame_ids)))
    if len(clip) != expected:
        raise ValueError(
            f"Aligned RGB length {len(clip)} != video_num_frames {expected}: {latent_path}"
        )
    return clip


implementation.read_video_window = aligned_video_window


if __name__ == "__main__":
    implementation.main()
