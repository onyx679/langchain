import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

import langchain_huggingface.embeddings.huggingface as hf_embeddings_module
from langchain_huggingface import HuggingFaceEmbeddings


def test_ipex_backend_rejects_optimum_intel_v2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that IPEX embeddings fail clearly with optimum-intel v2."""
    sentence_transformers = ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = MagicMock()
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)
    monkeypatch.setattr(
        hf_embeddings_module, "is_optimum_intel_available", lambda: True
    )
    monkeypatch.setattr(hf_embeddings_module, "is_ipex_available", lambda: True)
    monkeypatch.setattr(
        hf_embeddings_module,
        "is_optimum_intel_version",
        lambda operation, reference_version: (
            operation == ">=" and reference_version == "2.0"
        ),
    )

    with pytest.raises(ImportError, match=r"optimum-intel<2\.0"):
        HuggingFaceEmbeddings(model_kwargs={"backend": "ipex"})
