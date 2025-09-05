import logging
import os
import gc
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("pdfparser.faiss_service")


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_BASE_PATH = Path(os.environ.get("API_VOLUME_DIR", "/data/api"))


def _model_dir(model_name: str) -> Path:
    """Return filesystem path where the model is stored."""
    safe_name = model_name.replace("/", "_")
    return MODEL_BASE_PATH / safe_name


def ensure_model(model_name: str = MODEL_NAME) -> Path:
    """Ensure the SentenceTransformer model is stored on disk and return its path."""
    path = _model_dir(model_name)
    config_file = path / "config.json"
    if not config_file.exists():
        logger.info("Caching model '%s' to %s", model_name, path)
        model = SentenceTransformer(model_name)
        path.mkdir(parents=True, exist_ok=True)
        model.save(str(path))
        del model
        gc.collect()
        try:  # Free any GPU memory if torch is available
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return path


def build_faiss_index(
    texts: List[str],
    metadatas: List[Dict[str, Any]],
    model_name: str = MODEL_NAME,
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
    model_path = ensure_model(model_name)
    model = SentenceTransformer(str(model_path))
    embeddings = model.encode(texts, convert_to_numpy=True)
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
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
    model_name: str = MODEL_NAME,
    top_k: int = 5,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Search a FAISS index and return the top matches and metadata."""
    model_path = ensure_model(model_name)
    model = SentenceTransformer(str(model_path))
    query_vec = model.encode([query], convert_to_numpy=True)
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    _distances, indices = index.search(query_vec, top_k)
    results = [(texts[i], metadatas[i]) for i in indices[0]]

    return results
