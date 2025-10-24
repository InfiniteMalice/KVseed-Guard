"""Seed compilation tooling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from ..core.errors import VerificationError
from .sign_seed import load_signing_key, sign_seed_bundle


def _ensure_out(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _tensor_for_layer(layer_index: int, heads: int, seq_len: int, head_dim: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    base = torch.arange(heads * seq_len * head_dim, dtype=torch.float32).view(heads, seq_len, head_dim)
    key = (base + layer_index).to(dtype=dtype)
    value = torch.flip(base, dims=[-1]).add_(layer_index).to(dtype=dtype)
    return key, value


def _write_bin(path: Path, key: torch.Tensor, value: torch.Tensor, dtype: torch.dtype) -> None:
    with path.open("wb") as handle:
        for layer in range(key.shape[0]):
            handle.write(key[layer].contiguous().cpu().numpy().tobytes())
            handle.write(value[layer].contiguous().cpu().numpy().tobytes())


def _build_metadata(policy: Dict[str, Any], dims: Dict[str, Any]) -> Dict[str, Any]:
    metadata = {
        "version": policy.get("version", "1.0"),
        "model_build_hash": policy["model_build_hash"],
        "seq_len": int(dims["seq_len"]),
        "dtype": dims.get("dtype", "float32"),
        "rope_scaling": policy.get("rope_scaling"),
        "layers": dims["layers"],
        "insertion": policy.get("insertion", {}),
        "policy": policy.get("policy", {}),
        "signature": "",
    }
    if "verification_key" not in metadata["policy"]:
        raise VerificationError("Policy must include verification_key under policy section")
    return metadata


def compile_seed(policy_path: Path, dims_path: Path, out_path: Path, signing_key_path: Path | None) -> None:
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    dims = json.loads(dims_path.read_text(encoding="utf-8"))
    metadata = _build_metadata(policy, dims)
    dtype = metadata["dtype"].lower()
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype not in dtype_map:
        raise VerificationError(f"Unsupported dtype: {dtype}")
    torch_dtype = dtype_map[dtype]
    layers = metadata["layers"]
    seq_len = metadata["seq_len"]
    key_tensors = []
    value_tensors = []
    for spec in layers:
        heads = int(spec["heads"])
        head_dim = int(spec["head_dim"])
        layer_index = int(spec.get("layer", len(key_tensors)))
        key_layer, value_layer = _tensor_for_layer(layer_index, heads, seq_len, head_dim, torch_dtype)
        key_tensors.append(key_layer)
        value_tensors.append(value_layer)
    key = torch.stack(key_tensors, dim=0)
    value = torch.stack(value_tensors, dim=0)
    _ensure_out(out_path)
    bin_path = out_path / "seed.bin"
    _write_bin(bin_path, key, value, torch_dtype)
    json_path = out_path / "seed.json"
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if signing_key_path is not None:
        key_obj = load_signing_key(signing_key_path)
        sign_seed_bundle(out_path, key_obj)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compile kvseed-guard seed bundle")
    parser.add_argument("--policy", required=True, help="Policy YAML path")
    parser.add_argument("--dims", required=True, help="Model dimension JSON path")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--signing-key", required=False, help="Optional signing key path")
    args = parser.parse_args()
    compile_seed(Path(args.policy), Path(args.dims), Path(args.out), Path(args.signing_key) if args.signing_key else None)
    print("Seed compiled", args.out)


if __name__ == "__main__":
    main()
