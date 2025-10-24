"""llama.cpp runtime adapter."""

from __future__ import annotations

from typing import Any, Callable, Dict, MutableMapping, Optional

from ..core.errors import AdapterError
from ..core.typing import RuntimeSession


class LlamaCppSession(RuntimeSession):
    """Adapter for llama.cpp server bindings."""

    def __init__(self, context: Any) -> None:
        try:
            import llama_cpp  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise AdapterError("llama-cpp-python is not installed") from exc
        self.context = context
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
        if hasattr(self.context, "kv_cache") and hasattr(self.context.kv_cache, "inject"):
            self.context.kv_cache.inject(kv_objects, layer_map=layer_map)
        else:
            self.context.__dict__["_kv_seed"] = kv_objects

    def set_attn_mask(self, mask_spec: Any) -> None:
        self.context.__dict__["_kv_mask"] = mask_spec

    def register_logits_hook(self, hook_fn: Callable[[Any], Any]) -> None:
        self._logits_hook = hook_fn
        if hasattr(self.context, "set_logits_processor"):
            self.context.set_logits_processor(hook_fn)
        else:
            self.context.__dict__["_kvguard_logits_hook"] = hook_fn

    def apply_hook(self, logits: Any) -> Any:
        if self._logits_hook is None:
            return logits
        return self._logits_hook(logits)


__all__ = ["LlamaCppSession"]
