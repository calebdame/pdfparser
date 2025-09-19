import json
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


def update_document_status(
    document_id: Any, answer: Any, *, raw_json: bool = False
) -> None:
    client = get_client()
    if not client:
        return
    try:
        labels_value: Any = answer
        if raw_json and isinstance(answer, str):
            try:
                labels_value = json.loads(answer)
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "Failed to parse raw JSON labels for document %s; storing empty JSON",
                    document_id,
                )
                labels_value = {}

        if raw_json and not isinstance(labels_value, (dict, list)):
            logger.warning(
                "Raw JSON labels for document %s were %s; expected JSON object or array",
                document_id,
                type(labels_value).__name__,
            )
            labels_value = {}

        try:
            json.dumps(labels_value)
        except (TypeError, ValueError):
            logger.warning(
                "Labels for document %s not JSON serializable; storing empty JSON",
                document_id,
            )
            labels_value = {}
        resp = (
            client.table("ccr_documents")
            .update(
                {
                    "status": "reviewed",
                    "mock_data_processed": True,
                    "labels": labels_value,
                }
            )
            .eq("id", document_id)
            .execute()
        )
        logger.info("Supabase update response: %s", resp)
    except Exception as exc:
        logger.exception("Failed to update Supabase: %s", exc)
