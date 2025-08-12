# app.py
import os
import logging
from typing import List

import requests
from fastapi import FastAPI, Request, HTTPException
from pdf2image import convert_from_bytes
from PIL import Image

# Placeholder import for future OpenAI integration
# import openai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdfparser")


def process_pdf(file_path: str) -> List[Image.Image]:
    """Download a PDF, convert to images, and log the page count.

    Args:
        file_path: Path of the PDF within the storage bucket.

    Returns:
        List of PIL Image objects representing each PDF page.
    """
    base_url = os.environ.get("ENV_URL", "").rstrip("/")
    if not base_url:
        logger.warning("ENV_URL not configured; cannot download PDF")
        return []

    pdf_url = f"{base_url}/{file_path}"
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


def send_images_to_openai(images: List[Image.Image], question: str, batch_size: int = 20) -> None:
    """Send images to an OpenAI vision model in batches.

    This function is a placeholder for future implementation.
    """
    open_ai_key = os.environ.get("OPEN_AI_KEY")
    if not open_ai_key:
        logger.warning("OPEN_AI_KEY not configured; skipping OpenAI call")
        return

    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        logger.debug("Prepared batch of %d images for OpenAI", len(batch))
        # TODO: Implement OpenAI API call using open_ai_key
        pass

app = FastAPI()

@app.get("/")
async def root():
    logger.info("Root endpoint called")
    return {"status": "running"}

@app.post("/webhook")
async def webhook(request: Request):
    expected = os.environ.get("WEBHOOK_AUTH_TOKEN", "").strip()

    # Extract token from headers
    auth = request.headers.get("Authorization", "")
    x_token = request.headers.get("X-Webhook-Token", "").strip()

    if auth.startswith("Bearer "):
        received = auth.split("Bearer ", 1)[1].strip()
    else:
        received = x_token  # fallback to custom header

    # Enforce token if configured
    if expected:
        if not received:
            logger.warning("Missing auth token (Authorization or X-Webhook-Token)")
            raise HTTPException(status_code=403, detail="Missing auth token")
        if received != expected:
            logger.warning("Invalid auth token provided")
            raise HTTPException(status_code=403, detail="Invalid auth token")
    else:
        logger.warning("WEBHOOK_AUTH_TOKEN not set; skipping auth check")

    # Read payload (JSON first, then raw)
    try:
        payload = await request.json()
    except Exception:
        body = await request.body()
        payload = body.decode("utf-8") if isinstance(body, (bytes, bytearray)) else body

    logger.info("Received webhook payload: %s", payload)

    file_path = (
        payload.get("record", {}).get("file_path") if isinstance(payload, dict) else None
    )
    if file_path:
        images = process_pdf(file_path)
        if images:
            send_images_to_openai(images, question="Is this document valid?")

    return {"status": "received"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
