from __future__ import annotations

from typing import Any, Dict, MutableMapping, Optional

import torch

from kvseed_guard.core import api
from kvseed_guard.core.typing import RuntimeSession


class DummySession(RuntimeSession):
    def __init__(self) -> None:
        self.reserved = None
        self.injected: Dict[str, Any] | None = None
        self.mask = None

    def reserve_kv(self, slots: int) -> MutableMapping[str, Any]:
        self.reserved = slots
        return {}

    def inject_kv(self, kv_objects: Dict[str, Any], layer_map: Optional[Dict[int, list[int]]] = None) -> None:
        self.injected = kv_objects
        self.layer_map = layer_map

    def set_attn_mask(self, mask_spec: Any) -> None:
        self.mask = mask_spec

    def register_logits_hook(self, hook_fn):
        self.hook = hook_fn


def test_apply_seed_injects_shapes(seed_bundle) -> None:
    seed = api.load_seed("unit-build", str(seed_bundle))
    prepared = api.prepare_seed(seed, tokenizer=None)
    session = DummySession()
    api.apply_seed(session, prepared)
    assert session.reserved == seed.metadata.seq_len
    assert session.injected is not None
    assert session.injected["key"].shape == torch.Size([2, 2, 4, 4])
    assert session.mask["type"] == "prefix"
    assert len(session.mask["mask"]) == seed.metadata.seq_len
