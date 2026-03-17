"""Tests for model path resolution — config defaults should resolve correctly."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# Project root (where pyproject.toml lives)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestConfigDefaults:
    """Verify config default paths are relative and point to models/ dir."""

    def test_vlm_model_path_is_relative(self):
        from image_search_app.config import settings
        p = Path(settings.vlm_model_path)
        assert not p.is_absolute(), "vlm_model_path should be relative"
        assert "models" in str(p), "vlm_model_path should be under models/"

    def test_llm_model_path_is_relative(self):
        from image_search_app.config import settings
        p = Path(settings.llm_model_path)
        assert not p.is_absolute(), "llm_model_path should be relative"
        assert "models" in str(p), "llm_model_path should be under models/"

    def test_llm_models_dir_is_relative(self):
        from image_search_app.config import settings
        p = Path(settings.llm_models_dir)
        assert not p.is_absolute(), "llm_models_dir should be relative"

    def test_text_embedding_model_dir_is_relative(self):
        from image_search_app.config import settings
        p = Path(settings.text_embedding_model_dir)
        assert not p.is_absolute(), "text_embedding_model_dir should be relative"
        assert "models" in str(p)

    def test_face_models_dir_is_relative(self):
        from image_search_app.config import settings
        p = Path(settings.face_models_dir)
        assert not p.is_absolute(), "face_models_dir should be relative"

    def test_vlm_model_path_resolves(self):
        from image_search_app.config import settings
        resolved = _PROJECT_ROOT / settings.vlm_model_path
        assert resolved.exists(), f"VLM model path does not exist: {resolved}"

    def test_llm_model_path_resolves(self):
        from image_search_app.config import settings
        resolved = _PROJECT_ROOT / settings.llm_model_path
        assert resolved.exists(), f"LLM model path does not exist: {resolved}"

    def test_embedding_model_dir_resolves(self):
        from image_search_app.config import settings
        resolved = _PROJECT_ROOT / settings.text_embedding_model_dir
        assert resolved.exists(), f"Embedding model dir does not exist: {resolved}"

    def test_face_detection_model_resolves(self):
        from image_search_app.config import settings
        resolved = _PROJECT_ROOT / settings.face_models_dir / settings.face_detection_model
        assert resolved.exists(), f"Face detection model does not exist: {resolved}"


class TestScanAvailableModels:
    """Verify scan_available_models finds LLM models under models/ dir."""

    def test_finds_qwen3(self):
        from image_search_app.tools.llm import scan_available_models
        models = scan_available_models()
        assert any("Qwen3" in m for m in models), f"Expected Qwen3 model in {models}"

    def test_returns_list(self):
        from image_search_app.tools.llm import scan_available_models
        models = scan_available_models()
        assert isinstance(models, list)

    def test_empty_for_nonexistent_dir(self):
        from image_search_app.tools.llm import scan_available_models
        models = scan_available_models("/nonexistent/path")
        assert models == []


class TestLLMServicePathResolution:
    """Verify LLMService.load resolves relative paths."""

    def test_load_resolves_relative_path(self):
        from image_search_app.tools.llm import LLMService, _PROJECT_ROOT

        service = LLMService()

        # Mock openvino_genai to avoid actually loading
        mock_pipeline = MagicMock()
        with patch.dict("sys.modules", {"openvino_genai": MagicMock()}):
            import sys
            mock_ov = sys.modules["openvino_genai"]
            mock_ov.LLMPipeline.return_value = mock_pipeline

            service.load(model_path="models/Qwen3-4B-Instruct-ov", device="CPU")

            # Verify the path was resolved to absolute
            call_args = mock_ov.LLMPipeline.call_args
            actual_path = call_args[0][0]
            assert Path(actual_path).is_absolute(), f"Expected absolute path, got {actual_path}"
            assert "models" in actual_path
            assert "Qwen3-4B-Instruct-ov" in actual_path

        service._pipeline = None  # cleanup


class TestCaptionerPathResolution:
    """Verify Captioner._load resolves relative VLM model path."""

    def test_load_resolves_relative_path(self):
        from image_search_app.ingestion.captioner import Captioner, _PROJECT_ROOT

        captioner = Captioner()

        with patch.dict("sys.modules", {"openvino_genai": MagicMock()}):
            import sys
            mock_ov = sys.modules["openvino_genai"]
            mock_pipeline = MagicMock()
            mock_ov.VLMPipeline.return_value = mock_pipeline

            captioner._load()

            call_args = mock_ov.VLMPipeline.call_args
            actual_path = call_args[0][0]
            assert Path(actual_path).is_absolute(), f"Expected absolute path, got {actual_path}"
            assert "models" in actual_path

        captioner._pipeline = None  # cleanup
