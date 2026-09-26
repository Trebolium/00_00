from openai import OpenAI

from config import LLM_MODEL, OPENROUTER_API_KEY

_client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")


def respond(history: list[dict]) -> str:
    """Send the running conversation to the LLM via OpenRouter and return its reply text."""
    completion = _client.chat.completions.create(model=LLM_MODEL, messages=history)
    return completion.choices[0].message.content.strip()
