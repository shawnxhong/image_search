"""Tests for Qwen2.5-VL image captioning."""

from unittest.mock import MagicMock, patch

from image_search_app.ingestion.captioner import (
    CAPTION_PROMPT,
    CAPTION_WITH_NAMES_PROMPT,
    CaptionResult,
    Captioner,
    _format_names,
    _load_image_as_tensor,
)


class TestFormatNames:
    def test_single_name(self):
        assert _format_names(["Alice"]) == "Alice"

    def test_two_names(self):
        assert _format_names(["Alice", "Bob"]) == "Alice, Bob"

    def test_three_names(self):
        assert _format_names(["A", "B", "C"]) == "A, B, C"

    def test_empty(self):
        assert _format_names([]) == ""


class TestLoadImageAsTensor:
    def test_small_image_no_resize(self, sample_image_no_exif):
        """A 200x200 image should not be resized."""
        tensor = _load_image_as_tensor(str(sample_image_no_exif))
        shape = tensor.shape
        assert shape[0] == 1          # batch
        assert shape[1] == 200        # height
        assert shape[2] == 200        # width
        assert shape[3] == 3          # channels

    def test_large_image_is_resized(self, tmp_path):
        """An image larger than MAX_IMAGE_PIXELS should be resized."""
        from PIL import Image
        big = Image.new("RGB", (4000, 3000))
        path = tmp_path / "big.jpg"
        big.save(str(path), "JPEG")

        tensor = _load_image_as_tensor(str(path))
        shape = tensor.shape
        assert shape[0] == 1
        # Should be resized: 4000*3000 = 12M >> 1M limit
        assert shape[1] * shape[2] <= 1024 * 1024 * 1.1  # some tolerance


class TestCaptioner:
    @patch("image_search_app.ingestion.captioner._load_image_as_tensor")
    def test_generate_returns_caption_result(self, mock_load_img):
        """generate() should return a CaptionResult with caption and confidence."""
        mock_load_img.return_value = MagicMock()

        captioner = Captioner()
        mock_pipeline = MagicMock()
        mock_pipeline.generate.return_value = "a cat sitting on a couch"
        captioner._pipeline = mock_pipeline

        result = captioner.generate("/fake/img.jpg")

        assert isinstance(result, CaptionResult)
        assert result.caption == "a cat sitting on a couch"
        assert result.confidence == 0.8

        # Verify the unconditional prompt was used
        call_args = mock_pipeline.generate.call_args
        assert call_args.args[0] == CAPTION_PROMPT

    @patch("image_search_app.ingestion.captioner._load_image_as_tensor")
    def test_generate_with_names_includes_names_in_prompt(self, mock_load_img):
        """generate_with_names() should format names into the prompt."""
        mock_load_img.return_value = MagicMock()

        captioner = Captioner()
        mock_pipeline = MagicMock()
        mock_pipeline.generate.return_value = "Alice and Bob are standing on a beach"
        captioner._pipeline = mock_pipeline

        result = captioner.generate_with_names("/fake/img.jpg", ["Alice", "Bob"])

        assert result.caption == "Alice and Bob are standing on a beach"
        assert result.confidence == 0.85

        call_args = mock_pipeline.generate.call_args
        prompt_used = call_args.args[0]
        assert "Alice, Bob" in prompt_used
        assert "Use their names" in prompt_used

    @patch("image_search_app.ingestion.captioner._load_image_as_tensor")
    def test_generate_strips_whitespace(self, mock_load_img):
        """Caption should be stripped of leading/trailing whitespace."""
        mock_load_img.return_value = MagicMock()

        captioner = Captioner()
        mock_pipeline = MagicMock()
        mock_pipeline.generate.return_value = "  a photo with spaces  \n"
        captioner._pipeline = mock_pipeline

        result = captioner.generate("/fake/img.jpg")
        assert result.caption == "a photo with spaces"

    @patch("image_search_app.ingestion.captioner._load_image_as_tensor")
    def test_generate_with_single_name(self, mock_load_img):
        """Single name should work correctly."""
        mock_load_img.return_value = MagicMock()

        captioner = Captioner()
        mock_pipeline = MagicMock()
        mock_pipeline.generate.return_value = "Doglashi is waving at the camera"
        captioner._pipeline = mock_pipeline

        result = captioner.generate_with_names("/fake/img.jpg", ["Doglashi"])

        call_args = mock_pipeline.generate.call_args
        prompt_used = call_args.args[0]
        assert "Doglashi" in prompt_used
        assert result.caption == "Doglashi is waving at the camera"
