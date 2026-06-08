"""
Integration tests — requires both the FastAPI server and Ollama to be running.

Start the server first:
    uvicorn app.main:app --reload

Then run:
    pytest tests/ -v
"""

import base64
import struct
import zlib

import httpx
import pytest

BASE_URL = "http://localhost:8000"
TIMEOUT = 60.0


def _make_test_png(width: int = 16, height: int = 16) -> str:
    """Generate a valid RGB PNG as a base64 string for image tests."""
    def chunk(ctype: bytes, data: bytes) -> bytes:
        c = ctype + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\xff\x80\x00" * width for _ in range(height))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode()


async def _collect_tokens(r: httpx.Response) -> list[str]:
    return [
        line[6:]
        async for line in r.aiter_lines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


@pytest.mark.asyncio
async def test_health():
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["ollama"] == "reachable"


@pytest.mark.asyncio
async def test_generate_streams_tokens():
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        async with client.stream(
            "POST", f"{BASE_URL}/generate",
            json={"prompt": "Say hello in one word."},
        ) as r:
            assert r.status_code == 200
            tokens = await _collect_tokens(r)
    assert len(tokens) > 0


@pytest.mark.asyncio
async def test_generate_with_temperature_and_max_tokens():
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        async with client.stream(
            "POST", f"{BASE_URL}/generate",
            json={"prompt": "Say hello.", "temperature": 0.1, "max_tokens": 20},
        ) as r:
            assert r.status_code == 200
            tokens = await _collect_tokens(r)
    assert len(tokens) > 0


@pytest.mark.asyncio
async def test_summarize_streams_tokens():
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        async with client.stream(
            "POST", f"{BASE_URL}/summarize",
            json={
                "text": (
                    "FastAPI is a modern Python web framework for building APIs. "
                    "It is fast, easy to use, and generates automatic interactive docs."
                )
            },
        ) as r:
            assert r.status_code == 200
            tokens = await _collect_tokens(r)
    assert len(tokens) > 0


@pytest.mark.asyncio
async def test_classify_streams_tokens():
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        async with client.stream(
            "POST", f"{BASE_URL}/classify",
            json={
                "text": "The stock market crashed today.",
                "categories": ["finance", "sports", "politics"],
            },
        ) as r:
            assert r.status_code == 200
            tokens = await _collect_tokens(r)
    assert len(tokens) > 0


@pytest.mark.asyncio
async def test_extract_keywords_streams_tokens():
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        async with client.stream(
            "POST", f"{BASE_URL}/extract-keywords",
            json={
                "text": "Machine learning models require large datasets and significant compute resources."
            },
        ) as r:
            assert r.status_code == 200
            tokens = await _collect_tokens(r)
    assert len(tokens) > 0


@pytest.mark.asyncio
async def test_describe_image_streams_tokens():
    image_b64 = _make_test_png()
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        async with client.stream(
            "POST", f"{BASE_URL}/describe-image",
            json={"image": image_b64, "prompt": "What colour is this image?"},
        ) as r:
            assert r.status_code == 200
            tokens = await _collect_tokens(r)
    assert len(tokens) > 0
