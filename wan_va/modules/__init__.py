# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
__all__ = [
    'load_transformer', 'load_text_encoder', 'load_tokenizer', 'load_vae',
    'WanVAEStreamingWrapper'
]


def __getattr__(name):
    if name in __all__:
        from .utils import (
            WanVAEStreamingWrapper,
            load_text_encoder,
            load_tokenizer,
            load_transformer,
            load_vae,
        )
        exports = {
            'load_transformer': load_transformer,
            'load_text_encoder': load_text_encoder,
            'load_tokenizer': load_tokenizer,
            'load_vae': load_vae,
            'WanVAEStreamingWrapper': WanVAEStreamingWrapper,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
