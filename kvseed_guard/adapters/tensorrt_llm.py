"""TensorRT-LLM runtime adapter."""

from __future__ import annotations

from typing import Any, Callable, Dict, MutableMapping, Optional

from ..core.errors import AdapterError
from ..core.typing import RuntimeSession


class TrtLlmSession(RuntimeSession):
    """Adapter for TensorRT-LLM runtime engines."""

    def __init__(self, runtime: Any) -> None:
        try:
            import tensorrt_llm  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise AdapterError("TensorRT-LLM is not installed") from exc
        self.runtime = runtime
        self._reservation: Dict[str, Any] = {}
        self._logits_hook: Optional[Callable[[Any], Any]] = None

    def reserve_kv(self, slots: int) -> MutableMapping[str, Any]:
        self._reservation = {"capacity": slots}
        return self._reservation

    def inject_kv(
        self,
        kv_objects: Dict[str, Any],
        layer_map: Optional[Dict[int, list[int]]] = None,
    ) -> None:
        self._reservation.update(kv_objects)
        injector = getattr(self.runtime, "inject_kv_cache", None)
        if callable(injector):
            injector(kv_objects, layer_map=layer_map)
        else:
            # CPU fallback for documentation/testing
            self.runtime.__dict__["_kv_seed"] = kv_objects

    def set_attn_mask(self, mask_spec: Any) -> None:
        self.runtime.__dict__["_kv_mask"] = mask_spec

    def register_logits_hook(self, hook_fn: Callable[[Any], Any]) -> None:
        self._logits_hook = hook_fn
        registrar = getattr(self.runtime, "register_logits_postprocessor", None)
        if callable(registrar):
            registrar(hook_fn)
        else:
            self.runtime.__dict__["_kvguard_logits_hook"] = hook_fn

    def apply_hook(self, logits: Any) -> Any:
        if self._logits_hook is None:
            return logits
        return self._logits_hook(logits)


__all__ = ["TrtLlmSession"]
