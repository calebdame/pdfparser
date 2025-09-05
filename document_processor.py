import os
import gc
import logging
from typing import Any, Dict, List, Tuple

from pdf_service import process_pdf
from ocr_service import ocr_images
from faiss_service import build_faiss_index, search_index

logger = logging.getLogger("pdfparser.document_processor")


def chunk_texts(texts: List[str], chunk_size: int, chunk_overlap: int) -> Tuple[List[str], List[Dict[str, int]]]:
    step = max(1, chunk_size - chunk_overlap)
    chunked_texts: List[str] = []
    metadatas: List[Dict[str, int]] = []
    for page_num, text in enumerate(texts, start=1):
        for chunk_idx, offset in enumerate(range(0, len(text), step), start=1):
            chunk = text[offset : offset + chunk_size]
            chunked_texts.append(chunk)
            metadatas.append({"page": page_num, "chunk": chunk_idx})
    return chunked_texts, metadatas


def process_document(file_path: str, document_id: Any) -> None:
    logger.info(
        "Begin processing document %s; process_only=%s, command='%s'",
        document_id
    )

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

        logger.info(
            "Executing search for command '%s' on document %s",
            document_id,
        )

    finally:
        gc.collect()
