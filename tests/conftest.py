from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pytest
from nacl.encoding import Base64Encoder
from nacl.signing import SigningKey

from kvseed_guard.seed.compile_seed import compile_seed


@pytest.fixture()
def signing_key(tmp_path: Path) -> Path:
    key = SigningKey.generate()
    key_path = tmp_path / "dev.ed25519"
    key_path.write_text(key.encode(encoder=Base64Encoder).decode("utf-8"), encoding="utf-8")
    return key_path


def _write_policy(tmp_path: Path, verification_key: str) -> Path:
    policy = {
        "version": "1.0",
        "model_build_hash": "unit-build",
        "rope_scaling": {"factor": 1.0, "position_offset": 0.0},
        "insertion": {"tokens": [1, 2, 3, 4]},
        "policy": {
            "verification_key": verification_key,
            "gate": {"unsafe_token_ids": [9], "refusal_token_id": 0},
        },
    }
    policy_path = tmp_path / "policy.yaml"
    import yaml

    policy_path.write_text(yaml.safe_dump(policy), encoding="utf-8")
    return policy_path


def _write_dims(tmp_path: Path) -> Path:
    dims = {
        "seq_len": 4,
        "dtype": "float32",
        "layers": [
            {"layer": 0, "heads": 2, "head_dim": 4},
            {"layer": 1, "heads": 2, "head_dim": 4},
        ],
    }
    dims_path = tmp_path / "dims.json"
    dims_path.write_text(json.dumps(dims), encoding="utf-8")
    return dims_path


@pytest.fixture()
def seed_bundle(tmp_path: Path, signing_key: Path) -> Path:
    from nacl.signing import SigningKey

    key = SigningKey(signing_key.read_text(encoding="utf-8"), encoder=Base64Encoder)
    policy_path = _write_policy(tmp_path, key.verify_key.encode(encoder=Base64Encoder).decode("utf-8"))
    dims_path = _write_dims(tmp_path)
    out_path = tmp_path / "bundle"
    compile_seed(policy_path, dims_path, out_path, signing_key)
    return out_path
