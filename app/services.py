from typing import AsyncGenerator

from app.ollama_client import OllamaClient


async def _sse(token_gen: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
    async for token in token_gen:
        yield f"data: {token}\n\n"
    yield "data: [DONE]\n\n"


async def chat(client: OllamaClient, messages: list[dict], model: str | None = None, options: dict | None = None):
    async for chunk in _sse(client.chat_stream(messages, model, options)):
        yield chunk


async def generate(client: OllamaClient, prompt: str, images: list[str] | None = None, model: str | None = None, options: dict | None = None):
    async for chunk in _sse(client.generate_stream(prompt, model, images, options)):
        yield chunk


async def describe_image(client: OllamaClient, image: str, prompt: str = "Describe this image in detail.", model: str | None = None, options: dict | None = None):
    async for chunk in _sse(client.generate_stream(prompt, model, images=[image], options=options)):
        yield chunk


async def summarize(client: OllamaClient, text: str, model: str | None = None, options: dict | None = None):
    prompt = f"Summarize the following text concisely:\n\n{text}"
    async for chunk in _sse(client.generate_stream(prompt, model, options=options)):
        yield chunk


async def classify(client: OllamaClient, text: str, categories: list[str], model: str | None = None, options: dict | None = None):
    cats = ", ".join(categories)
    prompt = (
        f"Classify the following text into exactly one of these categories: {cats}.\n\n"
        f"Text: {text}\n\nRespond with only the category name, nothing else."
    )
    async for chunk in _sse(client.generate_stream(prompt, model, options=options)):
        yield chunk


async def extract_keywords(client: OllamaClient, text: str, model: str | None = None, options: dict | None = None):
    prompt = (
        f"Extract the most important keywords from the following text.\n"
        f"Return them as a comma-separated list with no explanation.\n\nText: {text}"
    )
    async for chunk in _sse(client.generate_stream(prompt, model, options=options)):
        yield chunk
