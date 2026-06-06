from typing import AsyncGenerator

from app.ollama_client import OllamaClient


async def _sse(token_gen: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
    async for token in token_gen:
        yield f"data: {token}\n\n"
    yield "data: [DONE]\n\n"


async def generate(client: OllamaClient, prompt: str, model: str | None = None):
    async for chunk in _sse(client.generate_stream(prompt, model)):
        yield chunk


async def summarize(client: OllamaClient, text: str, model: str | None = None):
    prompt = f"Summarize the following text concisely:\n\n{text}"
    async for chunk in _sse(client.generate_stream(prompt, model)):
        yield chunk


async def classify(
    client: OllamaClient,
    text: str,
    categories: list[str],
    model: str | None = None,
):
    cats = ", ".join(categories)
    prompt = (
        f"Classify the following text into exactly one of these categories: {cats}.\n\n"
        f"Text: {text}\n\n"
        f"Respond with only the category name, nothing else."
    )
    async for chunk in _sse(client.generate_stream(prompt, model)):
        yield chunk


async def extract_keywords(
    client: OllamaClient, text: str, model: str | None = None
):
    prompt = (
        f"Extract the most important keywords from the following text.\n"
        f"Return them as a comma-separated list with no explanation.\n\n"
        f"Text: {text}"
    )
    async for chunk in _sse(client.generate_stream(prompt, model)):
        yield chunk
