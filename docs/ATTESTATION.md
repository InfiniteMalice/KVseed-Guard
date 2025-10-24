# Attestation Reports

Attestation bundles capture how a protected deployment behaves under red-team prompts and
benign traffic. Each report includes:

- Schema version and timestamp.
- Model identifier and seed path.
- Red-team success rate with per-case transcripts.
- Refusal-quality score, latency statistics, and KL divergence between seed and baseline
  logits.

## Workflow

1. Verify and apply the desired seed to your deployment.
2. Run `kvguard attest --model <build-hash> --seed-path <path> --out attestations`.
3. Optionally sign the resulting `attest.json` using `--signing-key` to produce
   `attest.signed.json`.

## Publishing guidance

- Publish the signed attestation bundle alongside your model release notes.
- Provide the verification key so auditors can check signatures.
- Re-run attestation after any model, seed, or policy change.

## Interpreting metrics

- **Red-team success rate** closer to 1.0 means more prompts were safely handled.
- **Refusal-quality score** reflects explicit policy language in refusals.
- **KL divergence** monitors drift between seed and baseline logits; large shifts warrant
  investigation.
- **Latency stats** ensure protections do not introduce unacceptable overhead.
