# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Small compatibility shims for diffusers imports."""


def patch_accelerate_init_empty_weights():
    """Expose accelerate.init_empty_weights for diffusers versions that expect it.

    Some environments install an accelerate build where init_empty_weights lives
    under accelerate.big_modeling but is not re-exported from accelerate.__init__.
    Diffusers imports it from the top level during module import.
    """
    try:
        import accelerate
    except ImportError:
        return

    if hasattr(accelerate, "init_empty_weights"):
        return

    try:
        from accelerate.big_modeling import init_empty_weights
    except ImportError:
        return

    accelerate.init_empty_weights = init_empty_weights
