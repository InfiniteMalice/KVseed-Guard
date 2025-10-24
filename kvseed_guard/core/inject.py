"""KV cache injection helpers."""

from __future__ import annotations

from typing import Any, Dict, MutableMapping

import torch

from .errors import InjectionError
from .typing import LayerMap, PreparedSeed, RuntimeSession


def compose_prefix_mask(mask: torch.Tensor, position_bias: torch.Tensor | None = None) -> Dict[str, Any]:
    """Return a runtime-agnostic prefix mask specification."""

    if mask.dim() != 1:
        raise InjectionError("Seed mask must be 1D over sequence length")
    spec: Dict[str, Any] = {
        "type": "prefix",
        "mask": mask.bool().tolist(),
    }
    if position_bias is not None:
        spec["position_bias"] = position_bias.tolist()
    return spec


def _validate_reservation(reservation: MutableMapping[str, Any]) -> None:
    if "capacity" not in reservation:
        return
    capacity = int(reservation["capacity"])
    if capacity <= 0:
        raise InjectionError("Runtime reservation reported zero capacity")


def _tensor_parallel_slices(prepared: PreparedSeed, layer_map: LayerMap) -> Dict[Any, Dict[str, torch.Tensor]]:
    result: Dict[Any, Dict[str, torch.Tensor]] = {}
    if layer_map is None:
        return result
    for shard, heads in layer_map.items():
        head_indices = torch.as_tensor(heads, dtype=torch.long)
        if head_indices.numel() == 0:
            raise InjectionError("Tensor-parallel head mapping cannot be empty")
        shard_key = torch.index_select(prepared.key, dim=1, index=head_indices)
        shard_value = torch.index_select(prepared.value, dim=1, index=head_indices)
        result[shard] = {
            "key": shard_key.contiguous(),
            "value": shard_value.contiguous(),
            "heads": [int(h) for h in head_indices.tolist()],
        }
    return result


def build_kv_payload(
    prepared: PreparedSeed,
    reservation: MutableMapping[str, Any],
    layer_map: LayerMap = None,
) -> Dict[str, Any]:
    """Merge seed tensors with runtime reservation metadata."""

    _validate_reservation(reservation)
    payload: Dict[str, Any] = dict(reservation)
    payload.update({
        "key": prepared.key,
        "value": prepared.value,
        "token_ids": prepared.token_ids,
    })
    slices = _tensor_parallel_slices(prepared, layer_map)
    if slices:
        payload["tensor_parallel"] = slices
    return payload


def inject_prepared_seed(
    session: RuntimeSession,
    prepared: PreparedSeed,
    reservation: MutableMapping[str, Any],
    layer_map: LayerMap = None,
) -> None:
    """Perform the runtime injection sequence."""

    payload = build_kv_payload(prepared, reservation, layer_map)
    session.inject_kv(payload, layer_map=layer_map)
