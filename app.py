import os
import logging
from typing import Any

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks

from pdf_service import process_pdf
from openai_service import send_images_to_openai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdfparser")

app = FastAPI()


@app.get("/")
async def root():
    logger.info("Root endpoint called")
    return {"status": "running"}


def _process_document(file_path: str, command: str, document_id: Any) -> None:
    """Convert the PDF to images and send them to OpenAI."""
    images = process_pdf(file_path)
    if images:
        send_images_to_openai(images, command=command, document_id=document_id)


@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
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
    status = str(record.get("status", "")).lower()

    if status in {"reviewed", "processed"} or record.get("mock_data_processed"):
        logger.info("Skipping document %s with status '%s'", document_id, status)
        return {"status": "skipped"}

    command = os.environ.get("COMMAND", "").strip()
    if file_path and command:
        background_tasks.add_task(_process_document, file_path, command, document_id)
        return {"status": "queued"}

    logger.warning("Missing file_path or COMMAND; nothing queued")
    return {"status": "ignored"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
