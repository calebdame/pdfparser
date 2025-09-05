import os
import logging
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger("pdfparser.pdf_service")


def process_pdf(file_path: str) -> List["Image.Image"]:
    """Download a PDF, convert to images, and log the page count.

    Args:
        file_path: Either an absolute URL to the PDF or the path within the
            storage bucket (combined with ``ENV_URL``).

    Returns:
        List of PIL Image objects representing each PDF page.
    """
    import requests
    from pdf2image import convert_from_bytes
    from PIL import Image

    base_url = os.environ.get("ENV_URL", "").rstrip("/")
    if file_path.startswith(("http://", "https://")):
        pdf_url = file_path
    elif base_url:
        pdf_url = f"{base_url}/{file_path}"
    else:
        logger.warning(
            "ENV_URL not configured and file_path is not an absolute URL; cannot download PDF"
        )
        return []

    logger.info("Downloading PDF from %s", pdf_url)

    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
    except Exception as exc:
        logger.exception("Failed to download PDF: %s", exc)
        return []

    try:
        images = convert_from_bytes(response.content)
    except Exception as exc:
        logger.exception("Failed to convert PDF to images: %s", exc)
        return []

    logger.info("PDF has %d pages", len(images))
    return images
