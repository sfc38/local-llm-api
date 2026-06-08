from typing import AsyncGenerator

from app.ollama_client import OllamaClient


async def _sse(token_gen: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
    async for token in token_gen:
        yield f"data: {token}\n\n"
    yield "data: [DONE]\n\n"


async def generate(
    client: OllamaClient,
    prompt: str,
    model: str | None = None,
    images: list[str] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
):
    async for chunk in _sse(
        client.generate_stream(prompt, model, images, temperature, max_tokens)
    ):
        yield chunk


async def describe_image(
    client: OllamaClient,
    image: str,
    prompt: str = "Describe this image in detail.",
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
):
    async for chunk in _sse(
        client.generate_stream(prompt, model, images=[image], temperature=temperature, max_tokens=max_tokens)
    ):
        yield chunk


async def summarize(
    client: OllamaClient,
    text: str,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
):
    prompt = f"Summarize the following text concisely:\n\n{text}"
    async for chunk in _sse(
        client.generate_stream(prompt, model, temperature=temperature, max_tokens=max_tokens)
    ):
        yield chunk


async def classify(
    client: OllamaClient,
    text: str,
    categories: list[str],
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
):
    cats = ", ".join(categories)
    prompt = (
        f"Classify the following text into exactly one of these categories: {cats}.\n\n"
        f"Text: {text}\n\n"
        f"Respond with only the category name, nothing else."
    )
    async for chunk in _sse(
        client.generate_stream(prompt, model, temperature=temperature, max_tokens=max_tokens)
    ):
        yield chunk


async def chat(
    client: OllamaClient,
    messages: list[dict],
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
):
    async for chunk in _sse(
        client.chat_stream(messages, model, temperature, max_tokens)
    ):
        yield chunk


async def extract_keywords(
    client: OllamaClient,
    text: str,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
):
    prompt = (
        f"Extract the most important keywords from the following text.\n"
        f"Return them as a comma-separated list with no explanation.\n\n"
        f"Text: {text}"
    )
    async for chunk in _sse(
        client.generate_stream(prompt, model, temperature=temperature, max_tokens=max_tokens)
    ):
        yield chunk
