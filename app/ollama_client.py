import json
from typing import AsyncGenerator

import httpx


class OllamaClient:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def _build_options(self, options: dict | None) -> dict:
        if not options:
            return {}
        mapping = {
            "temperature": "temperature",
            "max_tokens": "num_predict",
            "top_p": "top_p",
            "top_k": "top_k",
            "repeat_penalty": "repeat_penalty",
            "seed": "seed",
            "num_ctx": "num_ctx",
        }
        return {mapping[k]: v for k, v in options.items() if k in mapping and v is not None}

    async def is_reachable(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self.base_url}/api/tags")
                return r.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    async def get_models(self) -> list[str]:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{self.base_url}/api/tags")
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]

    async def generate_stream(
        self,
        prompt: str,
        model: str | None = None,
        images: list[str] | None = None,
        options: dict | None = None,
    ) -> AsyncGenerator[str, None]:
        target_model = model or self.model
        payload: dict = {"model": target_model, "prompt": prompt, "stream": True}
        if images:
            payload["images"] = images
        ollama_opts = self._build_options(options)
        if ollama_opts:
            payload["options"] = ollama_opts

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST", f"{self.base_url}/api/generate", json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        chunk = json.loads(line)
                        yield chunk.get("response", "")
                        if chunk.get("done"):
                            break

    async def chat_stream(
        self,
        messages: list[dict],
        model: str | None = None,
        options: dict | None = None,
    ) -> AsyncGenerator[str, None]:
        target_model = model or self.model
        ollama_messages = [
            {k: v for k, v in m.items() if k in ("role", "content", "images") and v is not None}
            for m in messages
        ]
        payload: dict = {"model": target_model, "messages": ollama_messages, "stream": True}
        ollama_opts = self._build_options(options)
        if ollama_opts:
            payload["options"] = ollama_opts

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST", f"{self.base_url}/api/chat", json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        chunk = json.loads(line)
                        yield chunk.get("message", {}).get("content", "")
                        if chunk.get("done"):
                            break
