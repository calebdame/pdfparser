import os
import gc
import logging
import re
from typing import Any, Dict, List, Tuple

from pdf_service import process_pdf
from ocr_service import ocr_images
from faiss_service import build_faiss_index
from question_answer_service import ask_questions_for_categories

logger = logging.getLogger("pdfparser.document_processor")


def chunk_texts(
    texts: List[str], chunk_size: int, chunk_overlap: int
) -> Tuple[List[str], List[Dict[str, int]]]:
    """Split texts into sentence-based chunks using token counts.

    Each page is segmented into sentences, then reassembled into chunks that
    do not exceed ``chunk_size`` tokens. Adjacent chunks share up to
    ``chunk_overlap`` tokens for context continuity. Metadata tracks the page
    number and sentence range for each chunk.
    """

    chunked_texts: List[str] = []
    metadatas: List[Dict[str, int]] = []

    for page_num, text in enumerate(texts, start=1):
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        token_counts = [len(s.split()) for s in sentences]
        start = 0
        while start < len(sentences):
            token_total = 0
            end = start
            while end < len(sentences) and token_total + token_counts[end] <= chunk_size:
                token_total += token_counts[end]
                end += 1

            chunk = " ".join(sentences[start:end])
            chunked_texts.append(chunk)
            metadatas.append(
                {"page": page_num, "start_sentence": start + 1, "end_sentence": end}
            )

            if end >= len(sentences):
                break

            overlap_tokens = 0
            overlap_start = end - 1
            while overlap_start >= start and overlap_tokens < chunk_overlap:
                overlap_tokens += token_counts[overlap_start]
                overlap_start -= 1
            start = overlap_start + 1

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
