"""Hugging Face generation loop adapter."""

from __future__ import annotations

from typing import Any, Callable, Dict, MutableMapping, Optional

from ..core.errors import AdapterError
from ..core.typing import RuntimeSession


class HfLoopSession(RuntimeSession):
    """Adapter for transformers models executed via generate loops."""

    def __init__(self, model: Any) -> None:
        try:
            import transformers  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise AdapterError("transformers is not installed") from exc
        if not hasattr(model, "forward"):
            raise AdapterError("Model does not expose forward()")
        self.model = model
        self._reservation: Dict[str, Any] = {}
        self._logits_hook: Optional[Callable[[Any], Any]] = None
        if not hasattr(model, "_kvguard_original_forward"):
            self._patch_forward()

    def _patch_forward(self) -> None:
        original_forward = self.model.forward

        def patched(*args: Any, **kwargs: Any) -> Any:
            outputs = original_forward(*args, **kwargs)
            if self._logits_hook is not None and hasattr(outputs, "logits"):
                logits = outputs.logits
                gated = self._logits_hook(logits)
                outputs.logits = gated
            return outputs

        self.model._kvguard_original_forward = original_forward
        self.model.forward = patched  # type: ignore[assignment]

    def reserve_kv(self, slots: int) -> MutableMapping[str, Any]:
        self._reservation = {"capacity": slots}
        return self._reservation

    def inject_kv(
        self,
        kv_objects: Dict[str, Any],
        layer_map: Optional[Dict[int, list[int]]] = None,
    ) -> None:
        self._reservation.update(kv_objects)
        self.model.__dict__["_kv_seed"] = kv_objects
        self.model.__dict__["_kv_layer_map"] = layer_map

    def set_attn_mask(self, mask_spec: Any) -> None:
        self.model.__dict__["_kv_mask"] = mask_spec

    def register_logits_hook(self, hook_fn: Callable[[Any], Any]) -> None:
        self._logits_hook = hook_fn

    def apply_hook(self, logits: Any) -> Any:
        if self._logits_hook is None:
            return logits
        return self._logits_hook(logits)


__all__ = ["HfLoopSession"]
