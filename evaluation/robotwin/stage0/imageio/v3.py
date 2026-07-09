"""PyAV-backed subset of imageio.v3 required by build_paired_latent_bank."""

from __future__ import annotations


def imiter(path, plugin=None):
    if plugin not in (None, "pyav"):
        raise ValueError(f"Only the pyav backend is supported, got {plugin!r}")
    try:
        import av
    except ImportError as exc:
        raise ImportError(
            "Video decoding requires PyAV; install the LingBot-VA/LeRobot runtime requirements"
        ) from exc
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            yield frame.to_ndarray(format="rgb24")
