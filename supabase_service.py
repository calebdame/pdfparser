import logging
import os
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from supabase import Client

logger = logging.getLogger("pdfparser.supabase_service")


def get_client() -> Optional["Client"]:
    from supabase import create_client

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        logger.warning("Supabase credentials not configured; skipping update")
        return None
    return create_client(url, key)


def update_document_status(document_id: Any, answer: str) -> None:
    client = get_client()
    if not client:
        return
    try:
        resp = (
            client.table("ccr_documents")
            .update(
                {
                    "status": "reviewed",
                    "mock_data_processed": True,
                    "labels": {"answer": answer},
                }
            )
            .eq("id", document_id)
            .execute()
        )
        logger.info("Supabase update response: %s", resp)
    except Exception as exc:
        logger.exception("Failed to update Supabase: %s", exc)
