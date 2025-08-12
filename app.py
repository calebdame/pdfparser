import os
import logging
from fastapi import FastAPI, Request, HTTPException
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdfparser")

app = FastAPI()

@app.get("/")
async def root():
    logger.info("Root endpoint called")
    return {"status": "running"}

@app.post("/webhook")
async def webhook(request: Request):
    expected_token = os.environ.get("WEBHOOK_AUTH_TOKEN")
    auth_header = request.headers.get("Authorization", "")
    token = auth_header.split("Bearer ")[-1].strip() if auth_header.startswith("Bearer ") else auth_header
    if expected_token and token != expected_token:
        logger.warning("Unauthorized webhook attempt")
        raise HTTPException(status_code=403, detail="Invalid auth token")
    if not expected_token:
        logger.warning("WEBHOOK_AUTH_TOKEN not set; skipping auth check")
    try:
        payload = await request.json()
    except Exception:
        body = await request.body()
        payload = body.decode("utf-8") if isinstance(body, bytes) else body
    logger.info("Received webhook payload: %s", payload)
    return {"status": "received"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
