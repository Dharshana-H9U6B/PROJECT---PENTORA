"""
Image input processor.

Validates and prepares uploaded images for analysis.
OCR is not performed locally — Gemini handles multimodal analysis directly.
This processor handles validation and preprocessing only.
"""

from PIL import Image
from typing import Optional
import io


MAX_IMAGE_SIZE_MB = 10
MAX_DIMENSION = 4096


def load_image_from_bytes(data: bytes) -> Image.Image:
    """Load a PIL Image from raw bytes."""
    return Image.open(io.BytesIO(data))


def validate_image(image: Image.Image) -> tuple[bool, str]:
    """
    Validate a PIL Image for use with analysis.

    Returns:
        (is_valid, error_message)
    """
    if image is None:
        return False, "No image provided."

    width, height = image.size
    if width < 50 or height < 50:
        return False, "Image is too small. Please upload a clearer screenshot."

    return True, ""


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Prepare image for Gemini analysis.

    - Converts to RGB (removes alpha channel if present)
    - Resizes if excessively large
    """
    # Convert RGBA → RGB
    if image.mode in ("RGBA", "P", "LA"):
        background = Image.new("RGB", image.size, (255, 255, 255))
        if image.mode == "P":
            image = image.convert("RGBA")
        if image.mode in ("RGBA", "LA"):
            background.paste(image, mask=image.split()[-1])
        image = background
    elif image.mode != "RGB":
        image = image.convert("RGB")

    # Resize if too large
    width, height = image.size
    if width > MAX_DIMENSION or height > MAX_DIMENSION:
        image.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)

    return image


def get_image_info(image: Image.Image) -> dict:
    """Return basic image metadata."""
    return {
        "size": image.size,
        "mode": image.mode,
        "format": getattr(image, "format", "unknown"),
    }
