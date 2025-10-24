.PHONY: setup lint test build seed.compile seed.verify serve.vllm attest

setup:
	pip install -e .[dev]

lint:
	ruff check .
	mypy kvseed_guard

test:
	pytest -q

build:
	python -m build

seed.compile:
	python -m kvseed_guard.seed.compile_seed --policy examples/policy.safe.yaml --dims examples/dims.safe.json --out artifacts/safe-v1 --signing-key examples/dev-seed.ed25519

seed.verify:
	python -m kvseed_guard.seed.verify_seed --build-hash safe-build --path artifacts/safe-v1

serve.vllm:
	python examples/server_vllm.py --model "$(MODEL)" --seed-path artifacts/safe-v1 --gate

attest:
	python -m kvseed_guard.attestation.attest --model my-model --seed-path artifacts/safe-v1 --out attestations
