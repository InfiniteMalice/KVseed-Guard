from __future__ import annotations

import torch

from kvseed_guard.core.logits_gate import build_logits_hook


def test_refusal_token_emitted() -> None:
    hook = build_logits_hook(
        {
            "unsafe_token_ids": [1],
            "refusal_token_id": 2,
            "max_resamples": 0,
            "threshold": 0.0,
        }
    )
    logits = torch.tensor([0.1, 5.0, 0.2])
    gated = hook(logits)
    assert torch.isinf(gated[0])
    assert gated[2] > 10.0


def test_resample_pass_through() -> None:
    hook = build_logits_hook(
        {
            "unsafe_token_ids": [],
            "threshold": 10.0,
            "max_resamples": 2,
            "resample_std": 0.1,
        }
    )
    logits = torch.tensor([0.1, 0.2, 0.3])
    gated = hook(logits)
    assert torch.allclose(gated, logits, atol=1.0)
