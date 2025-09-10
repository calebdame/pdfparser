import os
import logging
from typing import Any

import nltk
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks

from document_processor import process_document


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdfparser")

NLTK_DATA_DIR = os.environ.get("NLTK_DATA", os.path.join(os.path.dirname(__file__), "nltk_data"))


def ensure_nltk() -> None:
    """Ensure required NLTK resources are downloaded to the volume."""
    os.makedirs(NLTK_DATA_DIR, exist_ok=True)
    nltk.data.path.append(NLTK_DATA_DIR)
    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg, download_dir=NLTK_DATA_DIR, quiet=True)


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

    if status in {"reviewed", "processed"} or record.get("mock_data_processed"):
        logger.info("Skipping document %s with status '%s'", document_id, status)
        return {"status": "skipped"}

    if file_path:
        logger.info(
            "Queueing document %s for processing", document_id
        )
        background_tasks.add_task(
            process_document, file_path, document_id
        )
        return {"status": "queued"}

    logger.warning("Missing file_path; nothing queued")
    return {"status": "ignored"}


@app.on_event("startup")
async def startup_event() -> None:
    """Download required assets (NLTK and model) to the mounted volume."""
    ensure_nltk()
    preload = os.environ.get("PRELOAD_MODEL", "").lower()
    if preload in {"1", "true", "yes"}:
        from faiss_service import ensure_model

        ensure_model()


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
