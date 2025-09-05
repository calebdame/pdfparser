import logging
from typing import Any, Dict, List, Tuple

import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("pdfparser.faiss_service")


def build_faiss_index(
    texts: List[str],
    metadatas: List[Dict[str, Any]],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Tuple[faiss.Index, List[str], List[Dict[str, Any]]]:
    """Create a FAISS index from text snippets with attached metadata.

    Args:
        texts: List of strings to index.
        metadatas: Metadata dictionaries for each text block (e.g., page numbers).
        model_name: HuggingFace model used to compute embeddings.

    Returns:
        A tuple ``(index, texts, metadatas)`` so callers can search the FAISS
        index and reference the original text and associated metadata.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    logger.info("Built FAISS index with %d vectors", index.ntotal)
    return index, texts, metadatas


def search_index(
    query: str,
    index: faiss.Index,
    texts: List[str],
    metadatas: List[Dict[str, Any]],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: int = 5,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Search a FAISS index and return the top matches and metadata."""
    model = SentenceTransformer(model_name)
    query_vec = model.encode([query], convert_to_numpy=True)
    _distances, indices = index.search(query_vec, top_k)
    results = [(texts[i], metadatas[i]) for i in indices[0]]
    return results
