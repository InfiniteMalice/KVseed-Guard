"""Seed signing utilities."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Optional

from nacl.encoding import Base64Encoder
from nacl.signing import SigningKey

from ..core.errors import VerificationError
from .verify_seed import _combined_digest


def _load_metadata(path: Path) -> dict:
    json_path = path / "seed.json"
    if not json_path.exists():
        raise VerificationError(f"Missing seed metadata: {json_path}")
    return json.loads(json_path.read_text(encoding="utf-8"))


def sign_seed_bundle(path: Path, signing_key: SigningKey) -> str:
    """Sign a seed bundle and persist the signature."""

    metadata = _load_metadata(path)
    digest = _combined_digest(path, metadata)
    signature = signing_key.sign(digest).signature
    metadata["signature"] = base64.b64encode(signature).decode("utf-8")
    json_path = path / "seed.json"
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata["signature"]


def load_signing_key(path: Path) -> SigningKey:
    data = path.read_text(encoding="utf-8").strip()
    return SigningKey(data, encoder=Base64Encoder)


def generate_signing_key(path: Path) -> SigningKey:
    key = SigningKey.generate()
    path.write_text(key.encode(encoder=Base64Encoder).decode("utf-8"), encoding="utf-8")
    return key


def cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Sign kvseed-guard seed bundle")
    parser.add_argument("--path", required=True, help="Seed directory")
    parser.add_argument("--key", required=False, help="Signing key (base64 string or file)")
    parser.add_argument("--key-file", required=False, help="Signing key file path")
    parser.add_argument("--generate-key", required=False, help="Write new signing key to file")
    args = parser.parse_args()

    if args.generate_key:
        key_path = Path(args.generate_key)
        key = generate_signing_key(key_path)
        print(
            "Generated signing key; public key:",
            key.verify_key.encode(encoder=Base64Encoder).decode("utf-8"),
        )
        return

    path_obj = Path(args.path)
    key: Optional[SigningKey] = None
    if args.key_file:
        key = load_signing_key(Path(args.key_file))
    elif args.key:
        key = SigningKey(args.key, encoder=Base64Encoder)
    else:
        raise SystemExit("Signing key required via --key or --key-file")
    signature = sign_seed_bundle(path_obj, key)
    print("Seed signed:", signature)


if __name__ == "__main__":
    cli()
