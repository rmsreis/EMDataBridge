"""LLM adapter layer for EMDataBridge.

This module implements a pluggable LLM adapter API and a concrete adapter
that uses a local LLaMA model via the `llama-cpp-python` package.

Design notes:
- The adapter lazily imports `llama_cpp` to avoid requiring the dependency at
  package import time. If `llama_cpp` is not available, the adapter will raise
  a clear ImportError when used.
- Model path is configured via the `LLAMA_MODEL_PATH` environment variable or
  via the adapter constructor.
- A MockAdapter is provided for tests and CI where a real LLM isn't available.

Usage:
    from EMDataBridge.llm_adapter import get_llm_adapter
    adapter = get_llm_adapter(model_path='/path/to/ggml-model.bin')
    mapping = adapter.map_metadata(raw_metadata, prompt_template=..., max_tokens=256)

Security / privacy:
- Be careful sending full vendor metadata to third-party models; using a local
  LLaMA instance keeps data on-premise.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from functools import lru_cache
from typing import Any, Dict, Optional


class LLMAdapterError(Exception):
    pass


class MockAdapter:
    """Simple mock adapter used in tests.

    It returns a deterministic mapping suggestion derived from the input for
    predictable tests.
    """

    def __init__(self, **kwargs):
        self.name = "mock"

    def map_metadata(self, raw_metadata: Dict[str, Any], prompt_template: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        # Simple deterministic mapping: fingerprint some keys and echo back a
        # small canonical structure for tests.
        s = json.dumps(raw_metadata, sort_keys=True).encode('utf8')
        h = hashlib.sha1(s).hexdigest()
        return {
            "canonical": {
                "sample": {"name": raw_metadata.get("sample_name") or raw_metadata.get("name") or f"sample-{h[:8]}"},
                "instrument": {"model": raw_metadata.get("instrument_model")},
            },
            "provenance": {"adapter": "mock", "fingerprint": h}
        }


class LlamaAdapter:
    """Adapter that uses a local LLaMA model via `llama-cpp-python`.

    Requires: `llama-cpp-python` installed and a GGML model file (ggml-*.bin).
    Configure model path via `LLAMA_MODEL_PATH` env var or pass `model_path`.
    """

    def __init__(self, model_path: Optional[str] = None, n_ctx: int = 2048, temperature: float = 0.0):
        self.model_path = model_path or os.environ.get("LLAMA_MODEL_PATH")
        self.n_ctx = n_ctx
        self.temperature = temperature
        self._llama = None
        self._lock = threading.Lock()

    def _ensure_model_loaded(self):
        if self._llama is not None:
            return
        if not self.model_path:
            raise LLMAdapterError("No LLaMA model path configured. Set LLAMA_MODEL_PATH or pass model_path to LlamaAdapter.")
        try:
            # Lazy import
            from llama_cpp import Llama

            with self._lock:
                # Double-check under lock
                if self._llama is None:
                    self._llama = Llama(model_path=self.model_path, n_ctx=self.n_ctx)
        except Exception as e:
            raise LLMAdapterError(f"Failed to initialize local LLaMA model: {e}")

    @lru_cache(maxsize=1024)
    def _cached_prompt_result(self, prompt: str, max_tokens: int, stop: Optional[str]):
        # Note: model calls may not be pure functions if model RNG/temperature used
        # but caching by prompt is a helpful optimization for repeated prompts.
        self._ensure_model_loaded()
        # llama-cpp-python returns a dict with 'choices' list containing 'text'
        resp = self._llama(prompt=prompt, max_tokens=max_tokens, temperature=self.temperature, stop=stop)
        # Normalise return
        if isinstance(resp, dict) and "choices" in resp:
            return resp["choices"][0].get("text", "")
        # Fallback
        return str(resp)

    def map_metadata(self, raw_metadata: Dict[str, Any], prompt_template: Optional[str] = None, max_tokens: int = 256, stop: Optional[str] = None) -> Dict[str, Any]:
        """Map raw vendor metadata to a canonical structure using LLaMA.

        Args:
            raw_metadata: vendor metadata dict
            prompt_template: a prompt string; if None a default prompt is used
            max_tokens: max tokens to generate
            stop: optional stop sequence

        Returns:
            mapping dict with keys 'canonical' and 'provenance'
        """
        if prompt_template is None:
            prompt_template = (
                "You are a metadata normalizer. Given raw vendor metadata as JSON,"
                " map it to the canonical fields: sample.name, instrument.model, acquisition.date, "
                "image.width, image.height, image.pixel_size_nm, spectroscopy.type, spectroscopy.energy_range.\n\n"
                "INPUT_METADATA:\n{raw}\n\nOUTPUT as JSON with keys 'canonical' and 'notes'."
            )

        raw_json = json.dumps(raw_metadata, ensure_ascii=False)
        prompt = prompt_template.format(raw=raw_json)

        try:
            text = self._cached_prompt_result(prompt, max_tokens=max_tokens, stop=stop)
        except LLMAdapterError:
            raise
        except Exception as e:
            raise LLMAdapterError(f"LLaMA call failed: {e}")

        # Try to parse the model output as JSON; fall back to text capture.
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = {"canonical": {}, "notes": {"raw_response": text}}

        # Add provenance
        fingerprint = hashlib.sha1(raw_json.encode("utf8")).hexdigest()
        provenance = {"adapter": "llama-cpp", "model_path": self.model_path, "fingerprint": fingerprint}
        if isinstance(parsed, dict):
            parsed.setdefault("provenance", {}).update(provenance)
            return parsed
        return {"canonical": {}, "provenance": provenance, "notes": {"raw_response": str(parsed)}}


def get_llm_adapter(prefer: str = "llama", **kwargs):
    """Factory to return an adapter instance.

    prefer: 'llama' or 'mock'
    kwargs passed to adapter constructor
    """
    if prefer == "mock":
        return MockAdapter(**kwargs)
    # Default to llama adapter
    return LlamaAdapter(**kwargs)
