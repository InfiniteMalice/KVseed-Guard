"""Custom exceptions for kvseed-guard."""

from __future__ import annotations


class VerificationError(RuntimeError):
    """Raised when a seed or attestation fails verification."""


class AdapterError(RuntimeError):
    """Raised when a runtime adapter cannot complete its task."""


class InjectionError(RuntimeError):
    """Raised when KV injection fails."""


class CompatibilityError(RuntimeError):
    """Raised when model/runtime compatibility checks fail."""


__all__ = [
    "VerificationError",
    "AdapterError",
    "InjectionError",
    "CompatibilityError",
]
