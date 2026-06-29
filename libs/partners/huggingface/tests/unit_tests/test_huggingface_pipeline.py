import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

import langchain_huggingface.llms.huggingface_pipeline as hf_pipeline_module
from langchain_huggingface import HuggingFacePipeline

DEFAULT_MODEL_ID = "gpt2"


def test_initialization_default() -> None:
    """Test default initialization."""
    llm = HuggingFacePipeline()

    assert llm.model_id == DEFAULT_MODEL_ID


@patch("transformers.pipeline")
def test_initialization_with_pipeline(mock_pipeline: MagicMock) -> None:
    """Test initialization with a pipeline object."""
    mock_pipe = MagicMock()
    mock_pipe.model.name_or_path = "mock-model-id"
    mock_pipeline.return_value = mock_pipe

    llm = HuggingFacePipeline(pipeline=mock_pipe)

    assert llm.model_id == "mock-model-id"


@patch("transformers.AutoTokenizer.from_pretrained")
@patch("transformers.AutoModelForCausalLM.from_pretrained")
@patch("transformers.pipeline")
def test_initialization_with_from_model_id(
    mock_pipeline: MagicMock, mock_model: MagicMock, mock_tokenizer: MagicMock
) -> None:
    """Test initialization with the from_model_id method."""
    mock_tokenizer.return_value = MagicMock(pad_token_id=0)
    mock_model.return_value = MagicMock()

    mock_pipe = MagicMock()
    mock_pipe.task = "text-generation"
    mock_pipe.model = mock_model.return_value
    mock_pipeline.return_value = mock_pipe

    llm = HuggingFacePipeline.from_model_id(
        model_id="mock-model-id",
        task="text-generation",
    )

    assert llm.model_id == "mock-model-id"


def test_ipex_backend_rejects_optimum_intel_v2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that IPEX backend fails clearly with optimum-intel v2."""
    transformers = ModuleType("transformers")
    transformers.AutoModelForCausalLM = MagicMock()
    transformers.AutoModelForSeq2SeqLM = MagicMock()
    transformers.AutoTokenizer = MagicMock()
    transformers.AutoTokenizer.from_pretrained.return_value = MagicMock(pad_token_id=0)
    transformers.pipeline = MagicMock()
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(hf_pipeline_module, "is_optimum_intel_available", lambda: True)
    monkeypatch.setattr(hf_pipeline_module, "is_ipex_available", lambda: True)
    monkeypatch.setattr(
        hf_pipeline_module,
        "is_optimum_intel_version",
        lambda operation, reference_version: (
            operation == ">=" and reference_version == "2.0"
        ),
    )

    with pytest.raises(ImportError, match=r"optimum-intel<2\.0"):
        HuggingFacePipeline.from_model_id(
            model_id="mock-model-id",
            task="text-generation",
            backend="ipex",
        )
