import os
import logging
from typing import Any

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks

from document_processor import process_document
from faiss_service import ensure_model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdfparser")

app = FastAPI()


@app.get("/")
async def root():
    logger.info("Root endpoint called")
    return {"status": "running"}


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

    process_only = False
    if isinstance(payload, dict) and ("process_only" in payload or "process_only" in record):
        process_only = True
    logger.info("Parsed process_only=%s", process_only)

    if status in {"reviewed", "processed"} or record.get("mock_data_processed"):
        logger.info("Skipping document %s with status '%s'", document_id, status)
        return {"status": "skipped"}

    command = os.environ.get("COMMAND", "").strip()
    logger.info("Command from environment: '%s'", command)

    if file_path:
        logger.info(
            "Queueing document %s for processing", document_id
        )
        background_tasks.add_task(
            process_document, file_path, command, document_id, process_only
        )
        return {"status": "queued"}

    logger.warning("Missing file_path; nothing queued")
    return {"status": "ignored"}


@app.on_event("startup")
async def startup_event() -> None:
    """Pre-download the transformer model to the mounted volume."""
    ensure_model()


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
