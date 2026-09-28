"""AI provider wrapper. Only this file knows about Anthropic - swap it to use Gemini/OpenAI."""
import json
import os
import re
from pathlib import Path
from typing import Optional, Type, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

T = TypeVar("T", bound=BaseModel)

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5")
API_KEY = (os.getenv("ANTHROPIC_API_KEY") or "").strip()

# True -> real AI. False -> clearly labelled DEMO mode (heuristics + local question bank).
AI_ENABLED = bool(API_KEY)

_client = None


def _get_client():
    global _client
    if _client is None:
        import anthropic

        _client = anthropic.Anthropic(api_key=API_KEY, timeout=60.0, max_retries=2)
    return _client


class AIError(Exception):
    """Raised when the AI call fails or returns something we cannot validate."""


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found")
    return json.loads(text[start : end + 1])


def generate_json(system: str, user: str, schema: Type[T], max_tokens: int = 6000) -> T:
    """Call the model, parse its JSON reply and validate it against `schema` (one retry on bad output)."""
    import anthropic

    messages = [{"role": "user", "content": user}]
    last_err: Optional[Exception] = None
    for _ in range(2):
        try:
            resp = _get_client().messages.create(
                model=MODEL,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                output_config={"effort": "low"},  # fast responses for a live demo
            )
        except anthropic.AuthenticationError as e:
            raise AIError("The Anthropic API key was rejected. Check ANTHROPIC_API_KEY in .env.") from e
        except anthropic.RateLimitError as e:
            raise AIError("The AI service is rate-limited right now. Please retry in a few seconds.") from e
        except anthropic.APIConnectionError as e:
            raise AIError("Could not reach the AI service. Check your internet connection.") from e
        except anthropic.APIStatusError as e:
            raise AIError(f"The AI service returned an error ({e.status_code}).") from e

        if resp.stop_reason == "refusal":
            raise AIError("The AI declined this request.")
        text = "".join(b.text for b in resp.content if b.type == "text")
        try:
            return schema.model_validate(_extract_json(text))
        except (ValueError, ValidationError) as e:
            last_err = e
            messages = [
                {"role": "user", "content": user},
                {"role": "assistant", "content": text or "{}"},
                {
                    "role": "user",
                    "content": "That reply was not valid for the required JSON schema "
                    f"({str(e)[:300]}). Reply again with ONLY the corrected JSON object.",
                },
            ]
    raise AIError("The AI returned an invalid response. Please try again.") from last_err
