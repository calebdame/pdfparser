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
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        received = auth.split("Bearer ", 1)[1].strip()
    else:
        received = request.headers.get("X-Webhook-Token", "").strip()

    if expected and received != expected:
        logger.warning(
            "Unauthorized webhook: have_auth=%s have_x=%s",
            bool(auth), bool(request.headers.get("X-Webhook-Token"))
        )
        raise HTTPException(status_code=403, detail="Invalid auth token")
        
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
