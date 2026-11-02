import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _get_api_key(*keys: str) -> str:
    for key in keys:
        value = os.getenv(key, "").strip()
        if value:
            return value
    return ""


def get_gemini_api_key() -> str:
    return _get_api_key("GEMINI_API_KEY", "GOOGLE_API_KEY")


def validate_gemini_config() -> str:
    api_key = get_gemini_api_key()
    if not api_key:
        raise RuntimeError(
            "Gemini API key is not configured. Add GEMINI_API_KEY or "
            "GOOGLE_API_KEY to the project-root .env file."
        )
    return api_key


def get_xai_api_key() -> str:
    return _get_api_key("XAI_API_KEY")
