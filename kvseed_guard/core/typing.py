"""Shared typing helpers for kvseed-guard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Protocol

import torch


TensorDict = Dict[str, torch.Tensor]
LayerMap = Optional[Mapping[int, list[int]]]


@dataclass(slots=True)
class SeedMetadata:
    """Metadata for a seed artifact."""

    version: str
    model_build_hash: str
    seq_len: int
    dtype: str
    rope_scaling: Optional[Dict[str, Any]]
    insertion: Dict[str, Any]
    policy: Dict[str, Any]
    signature: str
    layers: list[Dict[str, Any]]


@dataclass(slots=True)
class Seed:
    """Parsed seed containing metadata and raw tensors."""

    metadata: SeedMetadata
    key: torch.Tensor
    value: torch.Tensor

    def to(self, device: torch.device | str) -> "Seed":
        return Seed(
            metadata=self.metadata,
            key=self.key.to(device=device),
            value=self.value.to(device=device),
        )


@dataclass(slots=True)
class PreparedSeed:
    """Seed ready for injection into a runtime session."""

    metadata: SeedMetadata
    key: torch.Tensor
    value: torch.Tensor
    mask: torch.Tensor
    token_ids: torch.Tensor


class RuntimeSession(Protocol):
    """Protocol describing the minimum runtime API kvseed-guard requires."""

    def reserve_kv(self, slots: int) -> MutableMapping[str, Any]:
        """Request runtime-specific KV reservation and return descriptor."""

    def inject_kv(
        self,
        kv_objects: Mapping[str, Any],
        layer_map: Mapping[int, list[int]] | None = None,
    ) -> None:
        """Inject tensors into the session's KV cache."""

    def set_attn_mask(self, mask_spec: Any) -> None:
        """Apply an attention mask specification for the prefill tokens."""

    def register_logits_hook(self, hook_fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """Register a hook to mutate logits before sampling."""


__all__ = [
    "TensorDict",
    "LayerMap",
    "SeedMetadata",
    "Seed",
    "PreparedSeed",
    "RuntimeSession",
]
