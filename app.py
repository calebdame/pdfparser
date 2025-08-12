import os
import logging
from typing import Any

from fastapi import FastAPI, Request, HTTPException

from pdf_service import process_pdf
from openai_service import send_images_to_openai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdfparser")

app = FastAPI()


@app.get("/")
async def root():
    logger.info("Root endpoint called")
    return {"status": "running"}


@app.post("/webhook")
async def webhook(request: Request):
    expected = os.environ.get("WEBHOOK_AUTH_TOKEN", "").strip()

    auth = request.headers.get("Authorization", "")
    x_token = request.headers.get("X-Webhook-Token", "").strip()

    if auth.startswith("Bearer "):
        received = auth.split("Bearer ", 1)[1].strip()
    else:
        received = x_token

    if expected:
        if not received:
            logger.warning("Missing auth token (Authorization or X-Webhook-Token)")
            raise HTTPException(status_code=403, detail="Missing auth token")
        if received != expected:
            logger.warning("Invalid auth token provided")
            raise HTTPException(status_code=403, detail="Invalid auth token")
    else:
        logger.warning("WEBHOOK_AUTH_TOKEN not set; skipping auth check")

    try:
        payload = await request.json()
    except Exception:
        body = await request.body()
        payload = body.decode("utf-8") if isinstance(body, (bytes, bytearray)) else body

    logger.info("Received webhook payload: %s", payload)

    record = payload.get("record", {}) if isinstance(payload, dict) else {}
    file_path = record.get("file_path")
    document_id: Any = record.get("id")

    if file_path:
        images = process_pdf(file_path)
        if images:
            command = os.environ.get("COMMAND", "").strip()
            send_images_to_openai(images, command=command, document_id=document_id)

    return {"status": "received"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
