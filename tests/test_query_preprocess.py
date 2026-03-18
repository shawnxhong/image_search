"""Tests for query preprocessing (translation)."""

from unittest.mock import MagicMock, patch

import pytest

from image_search_app.agent.langgraph_flow import _has_non_ascii, preprocess_query


class TestHasNonAscii:
    def test_ascii_only(self):
        assert _has_non_ascii("hello world") is False

    def test_ascii_with_numbers(self):
        assert _has_non_ascii("photos in 2024") is False

    def test_chinese_characters(self):
        assert _has_non_ascii("小丽在实验室") is True

    def test_mixed_chinese_english(self):
        assert _has_non_ascii("小丽in a lab") is True

    def test_empty_string(self):
        assert _has_non_ascii("") is False


class TestPreprocessQuery:
    def test_ascii_query_skips_llm(self):
        """ASCII-only queries should be returned unchanged without calling LLM."""
        with patch("image_search_app.agent.langgraph_flow.get_llm_service") as mock_get:
            result = preprocess_query("Tom at the beach")
            mock_get.assert_not_called()
            assert result == "Tom at the beach"

    def test_non_ascii_calls_llm(self):
        """Non-ASCII queries should trigger an LLM translation call."""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = "小丽 in a lab"

        with patch("image_search_app.agent.langgraph_flow.get_llm_service", return_value=mock_llm):
            result = preprocess_query("小丽在实验室")

        assert result == "小丽 in a lab"
        mock_llm.chat.assert_called_once()
        # Should be called without tools parameter
        call_kwargs = mock_llm.chat.call_args
        assert "tools" not in call_kwargs.kwargs

    def test_llm_failure_returns_original(self):
        """If LLM call fails, return the original query."""
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = RuntimeError("LLM not loaded")

        with patch("image_search_app.agent.langgraph_flow.get_llm_service", return_value=mock_llm):
            result = preprocess_query("小丽在实验室")

        assert result == "小丽在实验室"

    def test_empty_response_returns_original(self):
        """If LLM returns empty, return the original query."""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = ""

        with patch("image_search_app.agent.langgraph_flow.get_llm_service", return_value=mock_llm):
            result = preprocess_query("小丽在实验室")

        assert result == "小丽在实验室"

    def test_suspiciously_long_response_returns_original(self):
        """If LLM returns a very long response, return the original query."""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = "This is a very long explanation " * 20

        with patch("image_search_app.agent.langgraph_flow.get_llm_service", return_value=mock_llm):
            result = preprocess_query("小丽在实验室")

        assert result == "小丽在实验室"

    def test_strips_thinking_tags(self):
        """LLM response with <think> tags should be cleaned."""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = "<think>The user wants...</think>小丽 in a lab"

        with patch("image_search_app.agent.langgraph_flow.get_llm_service", return_value=mock_llm):
            result = preprocess_query("小丽在实验室")

        assert result == "小丽 in a lab"
