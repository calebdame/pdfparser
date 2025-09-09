import logging
import os
import gc
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
from multiprocessing import get_context

import faiss

logger = logging.getLogger("pdfparser.faiss_service")


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_BASE_PATH = Path(os.environ.get("API_VOLUME_DIR", "/data/api"))


def _model_dir(model_name: str) -> Path:
    """Return filesystem path where the model is stored."""
    safe_name = model_name.replace("/", "_")
    return MODEL_BASE_PATH / safe_name


def ensure_model(model_name: str = MODEL_NAME) -> Path:
    """Ensure the model files are stored on disk and return the path."""
    path = _model_dir(model_name)
    config_file = path / "config.json"
    if not config_file.exists():
        logger.info("Caching model '%s' to %s", model_name, path)
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=model_name,
            local_dir=path,
            local_dir_use_symlinks=False,
        )
    return path


def preprocess_text(text: str) -> str:
    """Normalize text before encoding."""
    text = text.lower().strip()
    # Remove control characters
    text = re.sub(r"[\x00-\x1f\x7f]", "", text)
    return text


def _encode_worker(texts: List[str], model_path: Path, queue) -> None:
    """Encode texts in a separate process and return embeddings via queue."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(model_path))
    embeddings = model.encode(texts, convert_to_numpy=True)
    queue.put(embeddings)


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

    texts = [preprocess_text(t) for t in texts]

    ctx = get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(target=_encode_worker, args=(texts, model_path, queue))
    process.start()
    embeddings = queue.get()
    process.join()
    queue.close()
    process.close()

    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    del embeddings
    gc.collect()
    logger.info("Built FAISS index (cosine) with %d vectors", index.ntotal)
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

    query = preprocess_text(query)

    ctx = get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(target=_encode_worker, args=([query], model_path, queue))
    process.start()
    query_vec = queue.get()
    process.join()
    queue.close()
    process.close()
    faiss.normalize_L2(query_vec)
    _distances, indices = index.search(query_vec, top_k)
    del query_vec
    gc.collect()
    results = [(texts[i], metadatas[i]) for i in indices[0]]

    return results
