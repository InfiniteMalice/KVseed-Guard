"""Public API for kvseed-guard."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .errors import VerificationError
from .inject import compose_prefix_mask, inject_prepared_seed
from .logits_gate import enable_gate
from .rope_bias import apply_rope_bias, compute_rope_bias
from .seed_loader import load_seed_artifacts, prepare_seed_for_injection
from .typing import PreparedSeed, RuntimeSession, Seed
from ..seed.verify_seed import verify_seed_bundle


def load_seed(build_hash: str, path: str) -> Seed:
    """Load and verify a seed bundle."""

    seed = load_seed_artifacts(path)
    if seed.metadata.model_build_hash != build_hash:
        raise VerificationError(
            f"Seed build hash mismatch: expected {build_hash}, got {seed.metadata.model_build_hash}"
        )
    verify_seed_bundle(Path(path), expected_build_hash=build_hash)
    return seed


def prepare_seed(seed: Seed, tokenizer: Any) -> PreparedSeed:
    """Prepare a seed for injection, applying optional RoPE bias."""

    prepared = prepare_seed_for_injection(seed, tokenizer)
    bias = compute_rope_bias(seed.metadata)
    if bias is not None:
        prepared.key = apply_rope_bias(prepared.key, bias)
        prepared.value = apply_rope_bias(prepared.value, bias)
    return prepared


def apply_seed(session: RuntimeSession, prepared: PreparedSeed) -> None:
    """Inject prepared seed tensors into a runtime session."""

    reservation = session.reserve_kv(prepared.metadata.seq_len)
    layer_map = None
    if isinstance(reservation, dict) and "layer_map" in reservation:
        layer_map = reservation["layer_map"]
    bias = compute_rope_bias(prepared.metadata)
    mask_spec = compose_prefix_mask(prepared.mask, bias)
    inject_prepared_seed(session, prepared, reservation, layer_map=layer_map)
    session.set_attn_mask(mask_spec)


def enable_logit_gate(session: RuntimeSession, gate_cfg: Dict[str, Any]) -> None:
    """Enable optional logit gating on the session."""

    enable_gate(session, gate_cfg)


__all__ = ["RuntimeSession", "load_seed", "prepare_seed", "apply_seed", "enable_logit_gate"]
