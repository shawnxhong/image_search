"""Integration test: VLM caption refinement with real model.

Requires the VLM model to be available. Skipped if model is not found.
Uses image_0019.jpg (man in blue shirt, shelves/books/monitor background).
"""

import pytest

from image_search_app.config import settings

# Skip if VLM model is not available
_MODEL_AVAILABLE = True
try:
    from pathlib import Path
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    model_path = Path(settings.vlm_model_path)
    if not model_path.is_absolute():
        model_path = _PROJECT_ROOT / model_path
    if not model_path.exists():
        _MODEL_AVAILABLE = False
except Exception:
    _MODEL_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _MODEL_AVAILABLE, reason="VLM model not available")

TEST_IMAGE = r"C:\Users\53422\Downloads\231207_FaceGrouping\image_0019.jpg"


class TestVLMCaptionRefine:
    """Test that the VLM captioner properly substitutes names in captions."""

    @pytest.fixture(autouse=True, scope="class")
    def captioner(self):
        """Load captioner once for all tests in this class."""
        from image_search_app.ingestion.captioner import Captioner
        cap = Captioner()
        cap._load()
        yield cap
        cap.unload()

    def test_unconditional_caption_describes_scene(self, captioner):
        """Baseline: unconditional caption should describe the man and scene."""
        result = captioner.generate(TEST_IMAGE)
        caption = result.caption.lower()
        # Should mention a man/person and some scene elements
        assert any(word in caption for word in ["man", "person", "he"]), (
            f"Unconditional caption should mention a person: {result.caption}"
        )

    def test_refine_substitutes_single_name(self, captioner):
        """Refinement with name 'Hank' should include 'Hank' in the caption."""
        # First get the original caption
        original = captioner.generate(TEST_IMAGE)
        original_caption = original.caption

        # Refine with name
        refined = captioner.generate_with_names(
            TEST_IMAGE, ["Hank"], original_caption=original_caption,
        )

        assert "Hank" in refined.caption, (
            f"Refined caption should contain 'Hank' but got: {refined.caption}"
        )
        # Should preserve some scene details from original
        assert len(refined.caption) > 10, (
            f"Refined caption too short, may have lost details: {refined.caption}"
        )

    def test_refine_preserves_scene_details(self, captioner):
        """Refined caption should keep scene details from the original."""
        original = captioner.generate(TEST_IMAGE)
        original_caption = original.caption

        refined = captioner.generate_with_names(
            TEST_IMAGE, ["Hank"], original_caption=original_caption,
        )

        # The refined caption should be roughly similar length to original
        # (not drastically shorter, which would indicate detail loss)
        assert len(refined.caption) >= len(original_caption) * 0.5, (
            f"Refined caption lost too many details.\n"
            f"Original ({len(original_caption)} chars): {original_caption}\n"
            f"Refined  ({len(refined.caption)} chars): {refined.caption}"
        )

    def test_refine_with_chinese_name(self, captioner):
        """Refinement should preserve non-ASCII names exactly."""
        original = captioner.generate(TEST_IMAGE)
        original_caption = original.caption

        refined = captioner.generate_with_names(
            TEST_IMAGE, ["\u72d7\u5934\u841d\u8389"], original_caption=original_caption,
        )

        assert "\u72d7\u5934\u841d\u8389" in refined.caption, (
            f"Refined caption should contain exact Chinese name but got: {refined.caption}"
        )
