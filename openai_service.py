import os
import logging
import base64
import io
import json
from typing import List, Any, TYPE_CHECKING

from supabase_service import update_document_status

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger("pdfparser.openai_service")


def send_images_to_openai(
    images: List["Image.Image"],
    command: str,
    document_id: Any = None,
    batch_size: int = 20,
) -> None:
    """Send images to an OpenAI vision model and update Supabase.

    The default model is ``gpt-4o-mini``, a cost-effective option for
    document understanding tasks. Set the ``OPENAI_MODEL`` environment
    variable to override this choice.
    """

    from PIL import Image  # noqa: F401
    from openai import OpenAI
    import gc

    open_ai_key = os.environ.get("OPEN_AI_KEY")
    if not open_ai_key:
        logger.warning("OPEN_AI_KEY not configured; skipping OpenAI call")
        return

    if not command:
        logger.warning("COMMAND not configured; skipping OpenAI call")
        return

    client = OpenAI(api_key=open_ai_key)

    limited_images = images[:20]
    image_parts = []
    for img in limited_images:
        gray_img = img.convert("L")
        buffer = io.BytesIO()
        gray_img.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        image_parts.append({"type": "input_image", "image_url": f"data:image/png;base64,{b64}"})
        buffer.close()
        gray_img.close()
        img.close()

    try:
        response = client.responses.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            input=[
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": command}, *image_parts],
                }
            ],
        )
        answer = response.output_text
        logger.info("OpenAI response: %s", answer)
        if document_id is not None:
            update_document_status(document_id, answer)
    except Exception as exc:
        logger.exception("OpenAI API call failed: %s", exc)
    finally:
        gc.collect()

def answer_questions_with_context(
    context: List[str],
    questions: List[dict],
) -> tuple[str, dict]:
    """Send questions with context to OpenAI and return raw and parsed answers.

    Args:
        context: List of text snippets providing background information.
        questions: List of dictionaries with keys ``tag_term``, ``question``,
            and ``answers`` which is a list of possible options.

    Returns:
        A tuple of the raw text response from OpenAI and a dictionary mapping
        each ``tag_term`` to its selected answer.
    """
    from openai import OpenAI

    open_ai_key = os.environ.get("OPEN_AI_KEY")
    if not open_ai_key:
        logger.warning("OPEN_AI_KEY not configured; skipping OpenAI call")
        return "", {}

    client = OpenAI(api_key=open_ai_key)

    context_text = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(context))
    question_lines = []
    for i, q in enumerate(questions, start=1):
        options = ", ".join(q.get("answers", [])) or "open ended"
        question_lines.append(
            f"{i}. tag_term: {q['tag_term']}\n   question: {q['question']}\n   options: [{options}]"
        )
    questions_text = "\n".join(question_lines)

    prompt = (
        "You are an assistant answering HOA questions based on provided context from " 
        "documents scanned with OCR. Since it is parsed from OCR from, there may be "
        "typos, so use logic and strategy to understand the potential meaning of terms "
        "if one or more wrongs characters create nonsense words.\n\n"
        "For each question, it is extremely important to reply with a correctly formatted "
        "JSON object mapping the 'tag_term' to the "
        "best answer. If options are given, respond with one of them exactly.\n"
        "If a question has no options, provide a concise answer under 300 characters.\n\n"
        f"Context:\n{context_text}\n\nQuestions:\n{questions_text}\n\n"
        "Return JSON only - Nothing that will cause parsing errors.  Also avoid exotic "
        "symbols, characters, and emojis that would make standard JSON and text encoding "
        "errors.  Must be JSON parsable by Python's json package."
    )

    try:
        response = client.responses.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        )
        raw_text = response.output_text
        logger.info("OpenAI raw response: %s", raw_text)
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            logger.exception("Failed to parse OpenAI response as JSON")
            parsed = {}
        return raw_text, parsed
    except Exception as exc:
        logger.exception("OpenAI API call failed: %s", exc)
        return "", {}
