import numpy as np
from pathlib import Path

from EMDataBridge.standardizer import EMDataStandardizer


def test_standardizer_merges_mock_adapter(tmp_path):
    """Verify EMDataStandardizer merges LLM-proposed canonical fields and records provenance when using mock adapter."""
    raw_meta = {
        "sample_name": "TestSample",
        "instrument_model": "TEM-1000",
        "some_vendor_field": "value",
    }

    std = EMDataStandardizer(llm_backend="mock")

    # Inject a fake handler for a synthetic extension so we can call standardize
    def fake_handler(p: Path):
        data = np.zeros((2, 2), dtype=np.uint8)
        return data, raw_meta

    std.supported_formats[".fake"] = fake_handler

    out = std.standardize(tmp_path / "file.fake")

    assert "metadata" in out
    md = out["metadata"]

    # MockAdapter should provide a sample.name and an _llm_provenance entry
    assert "sample" in md and ("name" in md["sample"])
    assert "_llm_provenance" in md and isinstance(md["_llm_provenance"], dict)
