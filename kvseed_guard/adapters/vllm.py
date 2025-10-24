"""vLLM runtime adapter."""

from __future__ import annotations

from typing import Any, Callable, Dict, MutableMapping, Optional

from ..core.errors import AdapterError
from ..core.typing import RuntimeSession


class VllmSession(RuntimeSession):
    """Adapter around a vLLM engine instance."""

    def __init__(self, engine: Any) -> None:
        try:
            import vllm  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise AdapterError("vLLM is not installed") from exc
        self.engine = engine
        self._logits_hook: Optional[Callable[[Any], Any]] = None
        self._reservation: Dict[str, Any] = {}

    def reserve_kv(self, slots: int) -> MutableMapping[str, Any]:
        self._reservation = {"capacity": slots}
        return self._reservation

    def inject_kv(
        self,
        kv_objects: Dict[str, Any],
        layer_map: Optional[Dict[int, list[int]]] = None,
    ) -> None:
        self._reservation.update(kv_objects)
        injector = getattr(self.engine, "kv_cache_interface", None)
        if injector is not None and hasattr(injector, "inject_external"):
            injector.inject_external(kv_objects, layer_map=layer_map)
        else:
            self.engine.__dict__["_kv_seed"] = kv_objects

    def set_attn_mask(self, mask_spec: Any) -> None:
        self.engine.__dict__["_kv_mask"] = mask_spec

    def register_logits_hook(self, hook_fn: Callable[[Any], Any]) -> None:
        self._logits_hook = hook_fn
        sampler = getattr(self.engine, "add_logits_processor", None)
        if callable(sampler):
            sampler(hook_fn)
        else:
            self.engine.__dict__["_kvguard_logits_hook"] = hook_fn

    def apply_hook(self, logits: Any) -> Any:
        """Utility for tests or integrations without native hook support."""

        if self._logits_hook is None:
            return logits
        return self._logits_hook(logits)


__all__ = ["VllmSession"]
