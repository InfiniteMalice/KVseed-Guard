# Overview

kvseed-guard injects a signed prefix into the KV cache of supported LLM runtimes at the
start of each request. The seed nudges the model toward strong refusal behaviour while
keeping the runtime stateless and head-agnostic. Optional logit gating can resample or veto
unsafe completions before sampling occurs.

Key components:

- **Seed tooling** compiles deterministic KV tensors, signs them with Ed25519, and verifies
  the signature on every request.
- **Runtime adapters** provide a uniform interface for vLLM, llama.cpp, TensorRT-LLM, and
  Hugging Face loops without modifying model architectures.
- **Logit gate** applies configurable veto/resample policies using lightweight heuristics
  and optional MLP scoring.
- **Attestation** runs red-team prompts, calculates refusal-quality metrics, and emits a
  signed transparency report.

The system is designed to fail closed: if verification fails, injection is aborted and the
request is rejected. All actions are auditable via logs and signed artifacts.
