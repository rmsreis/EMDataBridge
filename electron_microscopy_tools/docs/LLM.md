# LLM integration (local LLaMA)

EMDataBridge includes a pluggable LLM adapter layer. This document explains how
to enable local LLaMA models and the available `llm_backend` options.

Backends
- `auto` (default): prefer a local LLaMA adapter when `LLAMA_MODEL_PATH` is set,
  otherwise no LLM is used.
- `llama`: use the local LLaMA adapter (requires `llama-cpp-python` and a GGML model file).
- `mock`: use a deterministic MockAdapter (useful for tests and CI).
- `none`: disable any LLM mapping.

Configuration
- Environment variable: `LLAMA_MODEL_PATH` — path to a GGML model (e.g. `ggml-model-q4_0.bin`).
- `llm_backend` constructor parameter: many API constructors accept `llm_backend`
  (e.g., `EMDataStandardizer`, `EMFormatTranslator`, `EMDataDiscovery`). You can
  also pass `llm_model_path` explicitly to override `LLAMA_MODEL_PATH`.

Example: enable LLaMA via env var

```bash
export LLAMA_MODEL_PATH=/models/ggml-model-q4_0.bin
python -c "from EMDataBridge.standardizer import EMDataStandardizer; s=EMDataStandardizer(llm_backend='auto'); print('created')"
```

Example: enable mock adapter (tests / CI)

```python
from EMDataBridge.standardizer import EMDataStandardizer
std = EMDataStandardizer(llm_backend='mock')  # deterministic, fast for tests
```

Notes
- The `LlamaAdapter` uses `llama-cpp-python` and loads the model lazily when
  first used. Ensure the host has sufficient memory for the chosen model.
- All LLM-provided mappings are merged into the metadata and provenance is
  recorded under `metadata['_llm_provenance']` for auditing.
