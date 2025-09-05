import logging
from typing import List

from PIL import Image
import pytesseract

logger = logging.getLogger("pdfparser.ocr_service")


def ocr_images(images: List[Image.Image]) -> List[str]:
    """Run OCR on a list of PIL images using Tesseract.

    Each image is converted to text with ``pytesseract``. Any failure on a
    page logs an exception and returns an empty string for that page so
    downstream processing can continue.
    """
    texts: List[str] = []
    for page_num, image in enumerate(images, start=1):
        try:
            text = pytesseract.image_to_string(image)
        except Exception as exc:  # pragma: no cover - logging only
            logger.exception("OCR failed on page %d: %s", page_num, exc)
            text = ""
        finally:
            image.close()
        texts.append(text)
    return texts
