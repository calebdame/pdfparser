import os
import logging
import base64
from typing import Any

import nltk
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, UploadFile

from document_processor import process_document, process_document_bytes


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


@app.post("/process")
async def process_endpoint(request: Request, background_tasks: BackgroundTasks):
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("file")
        document_id = form.get("document_id")
        if not isinstance(upload, UploadFile):
            raise HTTPException(status_code=400, detail="Missing file")
        pdf_bytes = await upload.read()
    else:
        try:
            payload = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON") from exc
        data = payload.get("data")
        document_id = payload.get("document_id")
        if not data:
            raise HTTPException(status_code=400, detail="Missing PDF data")
        try:
            pdf_bytes = base64.b64decode(data)
        except Exception as exc:
            logger.warning("Invalid base64 PDF data: %s", exc)
            raise HTTPException(status_code=400, detail="Invalid base64 data")

    if not document_id:
        raise HTTPException(status_code=400, detail="Missing document_id")

    background_tasks.add_task(process_document_bytes, pdf_bytes, document_id)
    return {"status": "queued"}


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
