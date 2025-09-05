import csv
import logging
from collections import defaultdict
from typing import Dict, List, Any

from faiss_service import search_index
from openai_service import answer_questions_with_context

logger = logging.getLogger("pdfparser.question_answer_service")


def load_questions(csv_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load questions grouped by Tag Category from a CSV file."""
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            answers = [
                row.get("Answer 1", ""),
                row.get("Answer 2", ""),
                row.get("Answer 3", ""),
                row.get("Answer 4", ""),
            ]
            answers = [a for a in answers if a]
            groups[row["Tag Category"]].append(
                {
                    "tag_term": row["Original Tags"],
                    "question": row["Question"],
                    "answers": answers,
                }
            )
    return groups


def ask_questions_for_categories(
    csv_path: str,
    index,
    texts: List[str],
    metadatas: List[Dict[str, Any]],
    top_k: int = 5,
) -> None:
    """Iterate through categories, query FAISS, and send questions to OpenAI."""
    categories = load_questions(csv_path)
    for category, questions in categories.items():
        logger.info("Processing category '%s'", category)
        results = search_index(category, index, texts, metadatas, top_k=top_k)
        context = [
            f"{text} (page {_meta.get('page')})" if _meta.get("page") else text
            for text, _meta in results
        ]
        raw, parsed = answer_questions_with_context(context, questions)
        logger.info("Raw response: %s", raw)
        logger.info("Parsed answers: %s", parsed)
