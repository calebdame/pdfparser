import logging
from typing import List, Tuple

import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("pdfparser.faiss_service")


def build_faiss_index(
    texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> Tuple[faiss.Index, List[str]]:
    """Create a FAISS index from text snippets.

    Args:
        texts: List of strings to index.
        model_name: HuggingFace model used to compute embeddings.

    Returns:
        A tuple of (FAISS index, texts). The text list is returned so it can
        be referenced when performing searches.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    logger.info("Built FAISS index with %d vectors", index.ntotal)
    return index, texts


def search_index(
    query: str,
    index: faiss.Index,
    texts: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: int = 5,
) -> List[str]:
    """Search a FAISS index and return the top ``top_k`` matching texts."""
    model = SentenceTransformer(model_name)
    query_vec = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)
    results = [texts[i] for i in indices[0]]
    return results
