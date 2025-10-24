# Security Model

## Threat model

kvseed-guard assumes the underlying model weights and runtime remain trustworthy but may
produce unsafe completions when prompted adversarially. Attackers may attempt to:

- Inject malicious seeds.
- Replay outdated or mismatched seeds.
- Bypass logit gating.
- Hide tampering within runtime adapters.

## Controls

- **Signature enforcement:** Seeds are signed with Ed25519. Verification errors raise a
  `VerificationError` and abort injection.
- **Version locking:** The `model_build_hash` ties a seed to a specific model build. Mismatched
  hashes fail closed.
- **Head agnosticism:** KV tensors are injected without referencing specific attention heads,
  ensuring compatibility across architectures and tensor-parallel sharding.
- **Transparent policies:** Policy metadata includes verification keys, gate defaults, and audit
  parameters. No hidden behaviour is allowed.

## Operational guidance

- Rotate signing keys regularly; publish the matching verification keys alongside releases.
- Store private keys offline and load them only when compiling or signing seeds.
- Log verification outcomes and gate decisions for forensic review.
- Use the attestation tooling to publish signed transparency reports after major model updates.

## Failure handling

- Seed verification failures must fail closed; do not fall back to unsigned operation.
- Runtime adapters should bubble up `AdapterError` instances when dependencies are missing or
  runtime APIs change.
- Logit gating defaults to pass-through behaviour if no rules trigger; refusal tokens are only
  emitted when veto conditions persist after resampling attempts.
