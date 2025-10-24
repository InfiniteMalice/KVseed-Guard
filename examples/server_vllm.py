"""Example vLLM server integration."""

from __future__ import annotations

import argparse

from kvseed_guard.adapters.vllm import VllmSession
from kvseed_guard.core import api


def main() -> None:
    parser = argparse.ArgumentParser(description="Example vLLM server using kvseed-guard")
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed-path", required=True)
    parser.add_argument("--build-hash", default="safe-build")
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()

    # In a production deployment, load a real vLLM engine.
    engine = type("Engine", (), {"add_logits_processor": lambda self, fn: None})()
    session = VllmSession(engine)
    seed = api.load_seed(args.build_hash, args.seed_path)
    prepared = api.prepare_seed(seed, tokenizer=None)
    api.apply_seed(session, prepared)
    if args.gate:
        api.enable_logit_gate(session, seed.metadata.policy.get("gate", {}))
    print("Seed injected; ready to handle requests")


if __name__ == "__main__":
    main()
