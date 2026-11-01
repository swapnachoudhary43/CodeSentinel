import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


def _get_required_api_key(*keys: str, provider: str) -> str:
    for key in keys:
        api_key = os.getenv(key, "").strip()
        if api_key:
            return api_key

    key_names = ", ".join(keys)
    raise RuntimeError(
        f"{provider} API key is missing from the project-root .env file. "
        f"Expected one of: {key_names}."
    )


def get_gemini_api_key() -> str:
    return _get_required_api_key(
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        provider="Gemini",
    )


def get_xai_api_key() -> str:
    return _get_required_api_key("XAI_API_KEY", provider="xAI")