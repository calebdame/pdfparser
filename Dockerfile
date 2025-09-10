FROM python:3.10-slim

# System deps for OCR/PDF + build toolchain (for regex) + FAISS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr libtesseract-dev poppler-utils \
    build-essential python3-dev pkg-config \
    libgomp1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV NLTK_DATA=/app/nltk_data \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt .

# (Helpful) Upgrade pip tooling to get better wheel resolution
RUN pip install --upgrade pip setuptools wheel

RUN pip install -r requirements.txt

COPY . .
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
