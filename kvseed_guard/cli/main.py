"""Typer CLI entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from ..core import api
from ..core.errors import AdapterError, VerificationError
from ..seed.verify_seed import verify_seed_bundle
from ..seed.compile_seed import compile_seed
from ..seed.sign_seed import load_signing_key
from ..attestation.attest import run_attestation, write_report, sign_report

app = typer.Typer(help="kvseed-guard command line interface")
seed_app = typer.Typer(help="Seed tooling")
app.add_typer(seed_app, name="seed")


@seed_app.command("verify")
def seed_verify(path: Path = typer.Option(...), build_hash: str = typer.Option(...)) -> None:
    """Verify a seed bundle."""

    try:
        verify_seed_bundle(path, expected_build_hash=build_hash)
    except VerificationError as exc:
        raise typer.Exit(code=1) from exc
    typer.echo("Seed verification succeeded")


@seed_app.command("compile")
def seed_compile(
    policy: Path = typer.Option(...),
    dims: Path = typer.Option(...),
    out: Path = typer.Option(...),
    signing_key: Optional[Path] = typer.Option(None, help="Optional signing key"),
) -> None:
    """Compile a seed bundle from policy and model dims."""

    compile_seed(policy, dims, out, signing_key)
    typer.echo(f"Seed compiled at {out}")


@app.command("serve")
def serve(
    backend: str = typer.Argument(..., help="Runtime backend (vllm|llamacpp|hf)"),
    model: str = typer.Option(...),
    seed_path: Path = typer.Option(...),
    build_hash: str = typer.Option(...),
    gate: bool = typer.Option(False, "--gate/--no-gate"),
) -> None:
    """Run a minimal serving loop for demonstration."""

    seed = api.load_seed(build_hash, str(seed_path))
    prepared = api.prepare_seed(seed, tokenizer=None)
    typer.echo(f"Loaded seed for build {build_hash} with seq_len={seed.metadata.seq_len}")
    if backend == "vllm":
        from ..adapters.vllm import VllmSession

        try:
            session = VllmSession(engine={"model": model})
        except AdapterError as exc:
            raise typer.Exit(code=1) from exc
    elif backend == "llamacpp":
        from ..adapters.llamacpp import LlamaCppSession

        try:
            session = LlamaCppSession(context={"model": model})
        except AdapterError as exc:
            raise typer.Exit(code=1) from exc
    elif backend == "hf":
        from ..adapters.hf_loop import HfLoopSession

        dummy_model = type("Dummy", (), {"forward": lambda self, *a, **k: None})()
        try:
            session = HfLoopSession(dummy_model)
        except AdapterError as exc:
            raise typer.Exit(code=1) from exc
    else:
        typer.echo(f"Unsupported backend: {backend}")
        raise typer.Exit(code=1)
    api.apply_seed(session, prepared)
    if gate:
        api.enable_logit_gate(session, seed.metadata.policy.get("gate", {}))
        typer.echo("Logit gate enabled")
    typer.echo("Seed applied; server is ready (demo mode)")


@app.command("attest")
def attest(
    model: str = typer.Option(...),
    seed_path: Path = typer.Option(...),
    out: Path = typer.Option(...),
    signing_key: Optional[Path] = typer.Option(None),
) -> None:
    """Run attestation and optionally sign the report."""

    report = run_attestation(model, seed_path)
    report_path = write_report(report, out)
    typer.echo(f"Attestation report written to {report_path}")
    if signing_key:
        key = load_signing_key(signing_key)
        signed_path = sign_report(report_path, key)
        typer.echo(f"Signed attestation report written to {signed_path}")


if __name__ == "__main__":
    app()
