from google import genai

from config import get_gemini_api_key


def main():
    client = genai.Client(api_key=get_gemini_api_key())

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=(
                "Reply with exactly this sentence: "
                "CodeSentinel Gemini connection successful."
            ),
        )

        text = getattr(response, "text", "").strip()

        if not text:
            print("ERROR: Gemini returned no text.")
            return

        print("SUCCESS: Gemini API responded.")
        print("Response:", text)

    except Exception as exc:
        print(
            f"ERROR: Gemini API request failed: "
            f"{type(exc).__name__}: {exc}"
        )


if __name__ == "__main__":
    main()