"""Logit gating utilities."""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Sequence

import torch

DEFAULT_UNSAFE_TOKEN_IDS = {50256, 0}
DEFAULT_REFUSAL_TOKEN_ID = 1


def _build_linear_scorer(config: Dict[str, object]) -> Callable[[torch.Tensor], torch.Tensor]:
    weights = torch.tensor(config.get("weights"), dtype=torch.float32)
    bias = float(config.get("bias", 0.0))

    def scorer(logits: torch.Tensor) -> torch.Tensor:
        flat = logits.view(-1)
        length = min(flat.numel(), weights.numel())
        score = torch.dot(flat[:length], weights[:length]) + bias
        return score

    return scorer


def _should_veto(
    logits: torch.Tensor,
    unsafe_ids: Sequence[int],
    threshold: float,
    scorer: Optional[Callable[[torch.Tensor], torch.Tensor]],
) -> bool:
    top_id = int(torch.argmax(logits).item())
    if top_id in unsafe_ids:
        return True
    if scorer is not None:
        score = float(scorer(logits).item())
        if score > threshold:
            return True
    return False


def build_logits_hook(gate_cfg: Dict[str, object]) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a hook that enforces a veto/resample policy."""

    unsafe_ids_cfg = gate_cfg.get("unsafe_token_ids", DEFAULT_UNSAFE_TOKEN_IDS)
    unsafe_ids = [int(x) for x in unsafe_ids_cfg]
    refusal_token_id = int(gate_cfg.get("refusal_token_id", DEFAULT_REFUSAL_TOKEN_ID))
    threshold = float(gate_cfg.get("threshold", 0.0))
    max_resamples = int(gate_cfg.get("max_resamples", 2))
    resample_std = float(gate_cfg.get("resample_std", 0.5))
    mlp_cfg = gate_cfg.get("mlp")
    scorer = _build_linear_scorer(mlp_cfg) if isinstance(mlp_cfg, dict) else None

    def hook(logits: torch.Tensor) -> torch.Tensor:
        working = logits.clone()
        attempts = 0
        while _should_veto(working, unsafe_ids, threshold, scorer):
            if attempts >= max_resamples:
                refusal = torch.full_like(working, fill_value=-math.inf)
                refusal[..., refusal_token_id] = gate_cfg.get("refusal_logit", 50.0)
                return refusal
            noise = torch.randn_like(working) * resample_std
            working = logits + noise
            attempts += 1
        return working

    return hook


def enable_gate(session, gate_cfg: Dict[str, object]) -> None:
    """Helper that registers the hook on a session."""

    hook = build_logits_hook(gate_cfg)
    session.register_logits_hook(hook)


__all__ = ["build_logits_hook", "enable_gate"]
