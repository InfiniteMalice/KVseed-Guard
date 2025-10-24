from __future__ import annotations

from kvseed_guard.core.seed_loader import load_seed_artifacts, prepare_seed_for_injection


def test_roundtrip_seed(seed_bundle) -> None:
    seed = load_seed_artifacts(seed_bundle)
    assert seed.key.shape == (2, 2, 4, 4)
    assert seed.value.shape == (2, 2, 4, 4)
    prepared = prepare_seed_for_injection(seed, tokenizer=None)
    assert prepared.token_ids.tolist() == [1, 2, 3, 4]
    assert prepared.mask.sum().item() == seed.metadata.seq_len
