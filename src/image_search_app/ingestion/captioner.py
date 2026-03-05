from dataclasses import dataclass


@dataclass
class CaptionResult:
    caption: str
    confidence: float


class Captioner:
    """Vision-language captioner adapter stub."""

    def generate(self, image_path: str) -> CaptionResult:
        return CaptionResult(
            caption=f"Placeholder caption for {image_path}",
            confidence=0.25,
        )
