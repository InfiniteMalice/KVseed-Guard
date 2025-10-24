# Integration Guide

## vLLM

```python
from kvseed_guard.adapters.vllm import VllmSession
from kvseed_guard.core import api

engine = ...  # vLLM engine
session = VllmSession(engine)
seed = api.load_seed(build_hash="model-build", path="/seeds/safe-v1")
prepared = api.prepare_seed(seed, tokenizer=engine.get_tokenizer())
api.apply_seed(session, prepared)
api.enable_logit_gate(session, seed.metadata.policy.get("gate", {}))
```

Ensure your engine exposes `kv_cache_interface.inject_external` or inspect the stored
`session.engine._kv_seed` for manual integration.

## llama.cpp

```python
from kvseed_guard.adapters.llamacpp import LlamaCppSession
from kvseed_guard.core import api

ctx = ...  # llama.cpp context from llama_cpp Python bindings
session = LlamaCppSession(ctx)
seed = api.load_seed("model-build", "/seeds/safe-v1")
prepared = api.prepare_seed(seed, tokenizer=ctx.tokenizer)
api.apply_seed(session, prepared)
```

If your build lacks a dedicated injection API, read the injected tensors from
`ctx._kv_seed` and apply them to the KV cache before generation.

## Hugging Face loop

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from kvseed_guard.adapters.hf_loop import HfLoopSession
from kvseed_guard.core import api

model = AutoModelForCausalLM.from_pretrained("model")
tokenizer = AutoTokenizer.from_pretrained("model")
session = HfLoopSession(model)
seed = api.load_seed("model-build", "/seeds/safe-v1")
prepared = api.prepare_seed(seed, tokenizer)
api.apply_seed(session, prepared)
```

The adapter patches `model.forward` to apply logit hooks. Inspect `model._kv_seed` and
`model._kv_mask` to integrate with custom generation loops.
