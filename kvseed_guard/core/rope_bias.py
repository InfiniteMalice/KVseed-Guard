"""RoPE bias utilities for seed tokens."""

from __future__ import annotations

from typing import Optional

import torch

from .typing import SeedMetadata


def compute_rope_bias(metadata: SeedMetadata) -> Optional[torch.Tensor]:
    """Return optional positional bias derived from seed metadata."""

    config = metadata.rope_scaling
    if not config:
        return None
    seq_len = metadata.seq_len
    offset = float(config.get("position_offset", 0.0))
    factor = float(config.get("factor", 1.0))
    base = torch.arange(seq_len, dtype=torch.float32)
    bias = base * factor + offset
    return bias


def apply_rope_bias(tensor: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
    """Apply a light-weight bias to K/V tensors for diagnostic parity."""

    if bias is None:
        return tensor
    while bias.dim() < tensor.dim():
        bias = bias.unsqueeze(-1)
    return tensor + bias.to(dtype=tensor.dtype)


__all__ = ["compute_rope_bias", "apply_rope_bias"]
