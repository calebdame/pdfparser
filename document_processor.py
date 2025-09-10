import os
import gc
import logging
from typing import Any, Dict, List, Tuple

import nltk
from nltk.tokenize import word_tokenize

NLTK_DATA_DIR = os.environ.get(
    "NLTK_DATA", os.path.join(os.path.dirname(__file__), "nltk_data")
)
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)
for pkg in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, download_dir=NLTK_DATA_DIR, quiet=True)

from pdf_service import process_pdf, process_pdf_bytes
from ocr_service import ocr_images
from faiss_service import build_faiss_index
from question_answer_service import ask_questions_for_categories

logger = logging.getLogger("pdfparser.document_processor")

def chunk_texts(texts: List[str], chunk_size: int, chunk_overlap: int) -> Tuple[List[str], List[Dict[str, int]]]:
    """Split texts into token-based chunks using NLTK.

    Each page is tokenized with :func:`nltk.word_tokenize`. Tokens are grouped into
    fixed-size windows with ``chunk_overlap`` overlap. Metadata tracks the page number
    and token index range for each chunk.
    """
    chunked_texts: List[str] = []
    metadatas: List[Dict[str, int]] = []

    step = chunk_size - chunk_overlap
    if step <= 0:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    for page_num, text in enumerate(texts, start=1):
        tokens = word_tokenize(text)
        for start in range(0, len(tokens), step):
            end = min(start + chunk_size, len(tokens))
            chunked_texts.append(" ".join(tokens[start:end]))
            metadatas.append({"page": page_num, "start_token": start, "end_token": end})

    return chunked_texts, metadatas

def process_document(file_path: str, document_id: Any) -> None:
    logger.info("Begin processing document %s", document_id)

    try:
        images = process_pdf(file_path)
        if not images:
            logger.warning(
                "PDF processing produced no images for document %s", document_id
            )
            return
        logger.info(
            "Converted document %s into %d images", document_id, len(images)
        )

        texts = ocr_images(images)
        del images
        gc.collect()
        if not texts:
            logger.warning(
                "No text extracted during OCR for document %s", document_id
            )
            return
        logger.info("Extracted OCR text for %d pages", len(texts))
        logger.info("First OCR text snippet: %s", texts[0][:200])

        chunk_size = int(os.environ.get("CHUNK_SIZE", 500))
        chunk_overlap = int(os.environ.get("CHUNK_OVERLAP", 100))
        logger.info(
            "Chunking texts with chunk_size=%d and chunk_overlap=%d",
            chunk_size,
            chunk_overlap,
        )
        chunked_texts, metadatas = chunk_texts(texts, chunk_size, chunk_overlap)
        logger.info("Generated %d chunks", len(chunked_texts))

        index, stored_texts, stored_metadatas = build_faiss_index(chunked_texts, metadatas)
        logger.info(
            "Built FAISS index for document %s with %d vectors",
            document_id,
            index.ntotal,
        )

        qa_csv = os.environ.get("QA_CSV_PATH", "hoana_questions.csv")
        qa_top_k = int(os.environ.get("QA_TOP_K", 5))
        ask_questions_for_categories(
            qa_csv, index, stored_texts, stored_metadatas, top_k=qa_top_k
        )

    finally:
        gc.collect()


def process_document_bytes(pdf_bytes: bytes, document_id: Any) -> None:
    logger.info("Begin processing document %s", document_id)

    try:
        images = process_pdf_bytes(pdf_bytes)
        if not images:
            logger.warning(
                "PDF processing produced no images for document %s", document_id
            )
            return
        logger.info(
            "Converted document %s into %d images", document_id, len(images)
        )

        texts = ocr_images(images)
        del images
        gc.collect()
        if not texts:
            logger.warning(
                "No text extracted during OCR for document %s", document_id
            )
            return
        logger.info("Extracted OCR text for %d pages", len(texts))
        logger.info("First OCR text snippet: %s", texts[0][:200])

        chunk_size = int(os.environ.get("CHUNK_SIZE", 500))
        chunk_overlap = int(os.environ.get("CHUNK_OVERLAP", 100))
        logger.info(
            "Chunking texts with chunk_size=%d and chunk_overlap=%d",
            chunk_size,
            chunk_overlap,
        )
        chunked_texts, metadatas = chunk_texts(texts, chunk_size, chunk_overlap)
        logger.info("Generated %d chunks", len(chunked_texts))

        index, stored_texts, stored_metadatas = build_faiss_index(chunked_texts, metadatas)
        logger.info(
            "Built FAISS index for document %s with %d vectors",
            document_id,
            index.ntotal,
        )

        qa_csv = os.environ.get("QA_CSV_PATH", "hoana_questions.csv")
        qa_top_k = int(os.environ.get("QA_TOP_K", 5))
        ask_questions_for_categories(
            qa_csv, index, stored_texts, stored_metadatas, top_k=qa_top_k
        )

    finally:
        gc.collect()
