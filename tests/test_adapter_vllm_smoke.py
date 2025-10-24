from __future__ import annotations

import pytest
import torch

from kvseed_guard.core import api


def test_vllm_adapter_smoke(seed_bundle) -> None:
    pytest.importorskip("vllm", reason="vLLM not installed")
    from kvseed_guard.adapters.vllm import VllmSession

    class StubEngine:
        def __init__(self) -> None:
            self.hook = None

        def add_logits_processor(self, fn):
            self.hook = fn

    engine = StubEngine()
    session = VllmSession(engine)
    seed = api.load_seed("unit-build", str(seed_bundle))
    prepared = api.prepare_seed(seed, tokenizer=None)
    api.apply_seed(session, prepared)
    api.enable_logit_gate(session, {"unsafe_token_ids": [1], "refusal_token_id": 2, "max_resamples": 0})
    logits = torch.tensor([0.1, 1.0, 0.3])
    gated = engine.hook(logits)
    assert torch.isinf(gated[0])
