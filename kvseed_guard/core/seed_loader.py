"""Seed loading utilities."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .errors import VerificationError
from .typing import PreparedSeed, Seed, SeedMetadata


_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _load_metadata(path: Path) -> SeedMetadata:
    json_path = path / "seed.json"
    if not json_path.exists():
        raise VerificationError(f"Missing seed metadata: {json_path}")
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    required_fields = {
        "version",
        "model_build_hash",
        "seq_len",
        "layers",
        "dtype",
        "insertion",
        "policy",
        "signature",
    }
    missing = required_fields - payload.keys()
    if missing:
        raise VerificationError(f"Seed metadata missing fields: {sorted(missing)}")
    dtype_name = payload["dtype"].lower()
    if dtype_name not in _DTYPE_MAP:
        raise VerificationError(f"Unsupported seed dtype: {dtype_name}")
    metadata = SeedMetadata(
        version=str(payload["version"]),
        model_build_hash=str(payload["model_build_hash"]),
        seq_len=int(payload["seq_len"]),
        dtype=dtype_name,
        rope_scaling=payload.get("rope_scaling"),
        insertion=payload["insertion"],
        policy=payload["policy"],
        signature=str(payload["signature"]),
        layers=[dict(layer) for layer in payload["layers"]],
    )
    return metadata


def _parse_bin(path: Path, metadata: SeedMetadata) -> tuple[torch.Tensor, torch.Tensor]:
    bin_path = path / "seed.bin"
    if not bin_path.exists():
        raise VerificationError(f"Missing seed tensor payload: {bin_path}")
    data = bin_path.read_bytes()
    dtype = _DTYPE_MAP[metadata.dtype]
    elem_size = torch.tensor([], dtype=dtype).element_size()
    seq_len = metadata.seq_len
    key_tensors = []
    value_tensors = []
    offset = 0
    for layer in metadata.layers:
        heads = int(layer.get("heads", 0))
        head_dim = int(layer.get("head_dim", 0))
        if heads <= 0 or head_dim <= 0:
            raise VerificationError("Invalid layer specification in seed metadata")
        layer_elems = heads * seq_len * head_dim
        bytes_required = layer_elems * elem_size
        total_required = 2 * bytes_required
        if offset + total_required > len(data):
            raise VerificationError("Seed binary payload truncated")
        k_buf = memoryview(data)[offset : offset + bytes_required]
        offset += bytes_required
        v_buf = memoryview(data)[offset : offset + bytes_required]
        offset += bytes_required
        key_tensor = torch.frombuffer(k_buf, dtype=dtype).clone()
        value_tensor = torch.frombuffer(v_buf, dtype=dtype).clone()
        shape = (heads, seq_len, head_dim)
        key_tensors.append(key_tensor.view(shape))
        value_tensors.append(value_tensor.view(shape))
    if offset != len(data):
        raise VerificationError("Seed binary payload contains unexpected trailing bytes")
    key = torch.stack(key_tensors, dim=0)  # [layers, heads, seq, dim]
    value = torch.stack(value_tensors, dim=0)
    return key, value


def load_seed_artifacts(path: str | os.PathLike[str]) -> Seed:
    """Load seed metadata and binary tensors from ``path``."""

    base = Path(path)
    metadata = _load_metadata(base)
    key, value = _parse_bin(base, metadata)
    return Seed(metadata=metadata, key=key, value=value)


def prepare_seed_for_injection(seed: Seed, tokenizer: Any) -> PreparedSeed:
    """Prepare seed tensors and mask for a tokenizer aware runtime."""

    seq_len = seed.metadata.seq_len
    insertion = seed.metadata.insertion
    tokens = insertion.get("tokens")
    if tokens is None:
        prompt = insertion.get("text")
        if prompt is None:
            raise VerificationError("Seed metadata missing tokens or text for insertion")
        if tokenizer is None:
            raise VerificationError("Tokenizer required to encode seed text")
        tokens = tokenizer.encode(prompt)
    token_ids = torch.tensor(tokens, dtype=torch.long)
    if token_ids.numel() != seq_len:
        raise VerificationError(
            f"Seed token length mismatch: expected {seq_len}, got {token_ids.numel()}"
        )
    mask = torch.ones((seq_len,), dtype=torch.bool)
    return PreparedSeed(
        metadata=seed.metadata,
        key=seed.key.clone(),
        value=seed.value.clone(),
        mask=mask,
        token_ids=token_ids,
    )


def seed_to_json(seed: Seed) -> str:
    """Utility for diagnostics."""

    payload = asdict(seed.metadata)
    payload["key_shape"] = list(seed.key.shape)
    payload["value_shape"] = list(seed.value.shape)
    return json.dumps(payload, indent=2, sort_keys=True)


__all__ = ["load_seed_artifacts", "prepare_seed_for_injection", "seed_to_json"]
