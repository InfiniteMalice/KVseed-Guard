# kvseed-guard

kvseed-guard is a production-ready middleware toolkit for injecting signed key-value (KV) cache seeds and optional logit gating into API-hosted large language models (LLMs). The project focuses on transparent, head-agnostic protection that reduces jailbreak success rates while improving refusal quality without modifying model architectures.

## Project intent

**Goal:** Reduce jailbreak success and increase refusal quality on API-served models by injecting a signed, version-locked KV seed at request start and (optionally) applying logit gating—with full transparency and auditability.

**Non-goals:** No backdoors, no hidden behaviour, no training/fine-tuning changes. No coupling to specific attention heads (MLA, Sparse, Mamba, etc.).

**Principles:** Head-agnostic; per-request stateless; version-locked to `model_build_hash`; signed seeds; fail-closed on verification errors; small latency overhead; adapters isolate runtime specifics.

## Features

- Signed, version-locked KV seeding compatible with multiple runtimes (vLLM, llama.cpp, TensorRT-LLM, Hugging Face loops).
- Optional logit gating with veto/resample policies and configurable refusal tokens.
- Seed compilation, signing, verification, and attestation tooling.
- Red-team suites and transparency metrics for continuous monitoring.
- Typer-powered CLI for seed verification, serving, and attestation workflows.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]
```

### Compile a sample seed

```bash
make seed.compile
```

This produces `artifacts/safe-v1/seed.json` and `seed.bin`. The seed includes a synthetic KV prefix for testing and is signed with a development Ed25519 key.

### Verify a seed

```bash
kvguard seed verify --build-hash safe-build --path artifacts/safe-v1
```

If verification fails, the command exits with an error to ensure fail-closed behaviour.

### Run the vLLM server example

```bash
make serve.vllm MODEL=TheBloke/TinyLLama-1.1B-Chat-v1.0
```

The example adapter reserves KV slots, injects the verified seed, and optionally enables the logit gate.

### Run attestation suite

```bash
kvguard attest --model my-model --seed-path artifacts/safe-v1 --out attestations
```

This executes the red-team suite, collects refusal-quality metrics, and writes a signed attestation bundle.

## Integration highlights

- Adapters isolate runtime-specific APIs and fall back gracefully when dependencies are missing.
- Logit gating is optional and integrates via runtime hooks without altering the base model.
- Seeds are per-request stateless and version-locked via `model_build_hash` to prevent drift.

## Transparency statement

kvseed-guard ships with comprehensive documentation, signed seed artifacts, and attestation workflows so integrators and auditors can review every control. The project rejects hidden controls and fails closed on verification issues, ensuring that protections are explicit, reproducible, and tamper-evident.

## License

kvseed-guard is released under the MIT License. See [LICENSE](LICENSE) for details.
