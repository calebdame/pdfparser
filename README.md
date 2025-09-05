# pdfparser

### High level Goal when done

This repo creates a webhook that accepts a payload from a Supabase instance, downloads the PDF links it shares, converts each page to an image, and processes the text locally. OpenAI calls are currently disabled; instead, the service logs OCR output and FAISS search results for testing purposes.

### Lower level beginning

Implement repos containing a server that can be used in railway.app

Set up a webhook to accept the payload from supabase

Set `WEBHOOK_AUTH_TOKEN` to require a matching `Authorization` header

Log all data received to console

### Configuration

* `OPEN_AI_KEY` – API key for OpenAI.
* `OPENAI_MODEL` – optional model override for image labeling. Defaults to
  `gpt-4o-mini`, which offers a lower-cost alternative while providing
  high quality results.

`record.file_path` in webhook payloads may be either a relative path or a full
URL. If it starts with `http://` or `https://`, the service downloads the PDF
from that URL directly; otherwise it is appended to the `ENV_URL` prefix.

### OCR and FAISS Indexing

PDF pages are now processed locally. The server converts each page to text using
[Tesseract](https://github.com/tesseract-ocr/tesseract) via `pytesseract` and
chunks the OCR output into segments (default 500 characters with 100 characters
of overlap; override with `CHUNK_SIZE` and `CHUNK_OVERLAP` environment
variables). These chunks are embedded with a small SentenceTransformer model
and stored in a local [FAISS](https://github.com/facebookresearch/faiss) index
alongside a metadata dictionary for each block (page and chunk numbers) so
queries can retrieve the most relevant snippets without sending images to 
external APIs.

### System dependencies

`pytesseract` only provides Python bindings; the Tesseract binary must be
present on the host. On Debian/Ubuntu-based images (including Railway's
default environment) install it with:

```sh
apt-get update && apt-get install -y tesseract-ocr
```

Ensure the `tesseract` command is on the `PATH` so the OCR service can invoke
it during PDF processing.

### Docker

A `Dockerfile` is included that installs Tesseract and Poppler utilities.
Build and run the server locally with:

```sh
docker build -t pdfparser .
docker run -p 8000:8000 pdfparser
```
