import os
import logging
import base64
import io
from typing import List, Any

from PIL import Image
from openai import OpenAI

from supabase_service import update_document_status

logger = logging.getLogger("pdfparser.openai_service")


def send_images_to_openai(
    images: List[Image.Image],
    command: str,
    document_id: Any = None,
    batch_size: int = 20,
) -> None:
    """Send images to an OpenAI vision model and update Supabase."""

    open_ai_key = os.environ.get("OPEN_AI_KEY")
    if not open_ai_key:
        logger.warning("OPEN_AI_KEY not configured; skipping OpenAI call")
        return

    if not command:
        logger.warning("COMMAND not configured; skipping OpenAI call")
        return

    client = OpenAI(api_key=open_ai_key)

    limited_images = images[:20]
    image_parts = []
    for img in limited_images:
        gray_img = img.convert("L")
        buffer = io.BytesIO()
        gray_img.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        image_parts.append({"type": "input_image", "image_url": f"data:image/png;base64,{b64}"})

    try:
        response = client.responses.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
            input=[
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": command}, *image_parts],
                }
            ],
        )
        answer = response.output_text
        logger.info("OpenAI response: %s", answer)
        if document_id is not None:
            update_document_status(document_id, answer)
    except Exception as exc:
        logger.exception("OpenAI API call failed: %s", exc)
