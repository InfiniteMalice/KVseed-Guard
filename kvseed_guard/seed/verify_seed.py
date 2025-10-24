"""Seed verification utilities."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Optional

from nacl import exceptions as nacl_exceptions
from nacl.encoding import Base64Encoder
from nacl.signing import VerifyKey

from ..core.errors import VerificationError


def _canonical_metadata(metadata: dict) -> bytes:
    canonical = dict(metadata)
    canonical["signature"] = ""
    return json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _combined_digest(path: Path, metadata: dict) -> bytes:
    json_digest = hashlib.sha256(_canonical_metadata(metadata)).digest()
    bin_path = path / "seed.bin"
    if not bin_path.exists():
        raise VerificationError(f"Missing seed tensor payload: {bin_path}")
    bin_digest = hashlib.sha256(bin_path.read_bytes()).digest()
    return json_digest + bin_digest


def verify_seed_bundle(
    path: Path,
    expected_build_hash: Optional[str] = None,
    public_key: Optional[str] = None,
) -> None:
    """Verify the signature and metadata for a seed bundle."""

    json_path = path / "seed.json"
    if not json_path.exists():
        raise VerificationError(f"Missing seed metadata: {json_path}")
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    if expected_build_hash is not None:
        build_hash = metadata.get("model_build_hash")
        if build_hash != expected_build_hash:
            raise VerificationError(
                f"Seed build hash mismatch: expected {expected_build_hash}, got {build_hash}"
            )
    signature_b64 = metadata.get("signature")
    if not signature_b64:
        raise VerificationError("Seed metadata missing signature")
    signature = base64.b64decode(signature_b64)
    digest = _combined_digest(path, metadata)
    if public_key is None:
        policy = metadata.get("policy") or {}
        public_key = policy.get("verification_key")
    if not public_key:
        raise VerificationError("Verification key not supplied")
    try:
        verify_key = VerifyKey(public_key, encoder=Base64Encoder)
        verify_key.verify(digest, signature)
    except (nacl_exceptions.BadSignatureError, ValueError) as exc:
        raise VerificationError("Seed signature validation failed") from exc


def main_cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Verify kvseed-guard seed bundle")
    parser.add_argument("--path", required=True, help="Seed directory path")
    parser.add_argument("--build-hash", required=False, help="Expected build hash")
    parser.add_argument("--public-key", required=False, help="Explicit verification key (base64)")
    args = parser.parse_args()
    verify_seed_bundle(Path(args.path), expected_build_hash=args.build_hash, public_key=args.public_key)
    print("Seed verification succeeded")


if __name__ == "__main__":
    main_cli()
