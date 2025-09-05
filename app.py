import os
import logging
from typing import Any

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdfparser")

app = FastAPI()


@app.get("/")
async def root():
    logger.info("Root endpoint called")
    return {"status": "running"}


def _process_document(file_path: str, command: str, document_id: Any) -> None:
    """Convert the PDF to images, run OCR, build a FAISS index, and log results."""
    from pdf_service import process_pdf
    from ocr_service import ocr_images
    from faiss_service import build_faiss_index  # , search_index

    images = process_pdf(file_path)
    if not images:
        return

    texts = ocr_images(images)
    if not texts:
        logger.warning("No text extracted during OCR for document %s", document_id)
        return

    logger.info("First OCR text snippet: %s", texts[0][:200])

    chunk_size = int(os.environ.get("CHUNK_SIZE", 500))
    chunk_overlap = int(os.environ.get("CHUNK_OVERLAP", 100))

    step = max(1, chunk_size - chunk_overlap)
    chunked_texts = []
    metadatas = []
    for page_num, text in enumerate(texts, start=1):
        for chunk_idx, offset in enumerate(range(0, len(text), step), start=1):
            chunk = text[offset : offset + chunk_size]
            chunked_texts.append(chunk)
            metadatas.append({"page": page_num, "chunk": chunk_idx})

    index, stored_texts, stored_metadatas = build_faiss_index(chunked_texts, metadatas)
    logger.info(
        "Built FAISS index for document %s with %d vectors",
        document_id,
        index.ntotal,
    )

    # if command:
    #     results = search_index(command, index, stored_texts, stored_metadatas, top_k=3)
    #     logger.info("Sample search results for '%s': %s", command, results)


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

    # command = os.environ.get("COMMAND", "").strip()
    if file_path:
        background_tasks.add_task(_process_document, file_path, "", document_id)
        return {"status": "queued"}
    # if file_path and command:
    #     background_tasks.add_task(_process_document, file_path, command, document_id)
    #     return {"status": "queued"}

    logger.warning("Missing file_path; nothing queued")
    return {"status": "ignored"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
