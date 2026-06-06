import json
from typing import AsyncGenerator

import httpx


class OllamaClient:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def is_reachable(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self.base_url}/api/tags")
                return r.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    async def generate_stream(
        self, prompt: str, model: str | None = None
    ) -> AsyncGenerator[str, None]:
        target_model = model or self.model
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json={"model": target_model, "prompt": prompt, "stream": True},
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        chunk = json.loads(line)
                        yield chunk.get("response", "")
                        if chunk.get("done"):
                            break
