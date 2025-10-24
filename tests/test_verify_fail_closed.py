from __future__ import annotations

import json
from pathlib import Path

import pytest

from kvseed_guard.core.api import load_seed
from kvseed_guard.core.errors import VerificationError
from kvseed_guard.seed.verify_seed import verify_seed_bundle


def test_build_hash_mismatch(seed_bundle: Path) -> None:
    with pytest.raises(VerificationError):
        load_seed("wrong-build", str(seed_bundle))


def test_signature_fail(seed_bundle: Path) -> None:
    meta_path = Path(seed_bundle) / "seed.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata["signature"] = "ZmFrZXNpZw=="  # base64 for "fakesig"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with pytest.raises(VerificationError):
        verify_seed_bundle(Path(seed_bundle), expected_build_hash="unit-build")
