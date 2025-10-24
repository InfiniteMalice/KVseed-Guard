# kvseed-guard Seed Format

Seeds consist of a metadata document (`seed.json`) and a binary tensor payload (`seed.bin`).
The pair is signed using Ed25519 to guarantee authenticity and version-lock seeds to a
`model_build_hash`.

## Metadata (`seed.json`)

The metadata is a UTF-8 encoded JSON object with the following fields:

| Field | Type | Description |
| ----- | ---- | ----------- |
| `version` | string | Semantic version of the seed schema. |
| `model_build_hash` | string | Build identifier of the model the seed was derived from. |
| `seq_len` | integer | Number of seed tokens. |
| `dtype` | string | Floating-point dtype for tensors (`float16`, `bfloat16`, `float32`). |
| `rope_scaling` | object or null | Optional RoPE bias info (`factor`, `position_offset`). |
| `layers` | array | Per-layer configuration objects. Each object includes `layer` (index), `heads`, `head_dim`. |
| `insertion` | object | Defines how to reconstruct the prefix tokens (`tokens` array or `text`). |
| `policy` | object | Free-form policy information. Must include `verification_key` (base64-encoded Ed25519 public key) and optional logit-gate defaults. |
| `signature` | string | Base64-encoded Ed25519 signature. Empty string prior to signing. |

### Canonicalisation

Prior to signing, serialise the metadata with the `signature` field set to the empty string using
`json.dumps(metadata, sort_keys=True, separators=(",", ":"))`. The SHA-256 digest of this canonical
serialisation is concatenated with the SHA-256 digest of the `seed.bin` payload; the combined digest is
then signed.

## Tensor payload (`seed.bin`)

`seed.bin` stores key and value tensors interleaved per layer. For each layer entry in `layers`:

1. Write a contiguous block for the key tensor shaped `[heads, seq_len, head_dim]`.
2. Write the corresponding value tensor block with the same shape.

All tensors use the dtype specified in `seed.json`. The binary file contains no headers or padding.
A SHA-256 checksum of the full binary payload is included implicitly via the combined digest during
signing.

## Signing workflow

1. Produce unsigned metadata and tensor payload (see `compile_seed.py`).
2. Canonicalise the metadata with an empty `signature` field.
3. Compute the combined digest: `sha256(canonical_json) || sha256(seed.bin)`.
4. Sign the digest using an Ed25519 private key.
5. Store the base64-encoded signature in `seed.json` and distribute the matching public key.

Verification recomputes the digest, checks the Ed25519 signature with the supplied public key, and
confirms that `model_build_hash` matches the runtime configuration. Verification failures must result
in a hard error (fail-closed).
