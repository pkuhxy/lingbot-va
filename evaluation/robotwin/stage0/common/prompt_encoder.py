from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F


class RobotwinPromptEncoder:
    def __init__(self, model_root, device="cuda:0", dtype=torch.bfloat16):
        from wan_va.modules.utils import load_text_encoder, load_tokenizer

        root = Path(model_root).expanduser().resolve()
        self.device, self.dtype = torch.device(device), dtype
        self.tokenizer = load_tokenizer(str(root / "tokenizer"))
        self.encoder = load_text_encoder(
            str(root / "text_encoder"), torch_dtype=dtype, torch_device=self.device
        )

    @torch.no_grad()
    def encode(self, prompt: str, max_length: int = 512) -> torch.Tensor:
        from diffusers.pipelines.wan.pipeline_wan import prompt_clean

        inputs = self.tokenizer(
            [prompt_clean(prompt)], padding="max_length", max_length=max_length,
            truncation=True, add_special_tokens=True, return_attention_mask=True,
            return_tensors="pt",
        )
        hidden = self.encoder(
            inputs.input_ids.to(self.device), inputs.attention_mask.to(self.device)
        ).last_hidden_state.to(self.dtype)
        length = int(inputs.attention_mask[0].sum())
        value = hidden[0, :length]
        return F.pad(value, (0, 0, 0, max_length - length)).cpu()
