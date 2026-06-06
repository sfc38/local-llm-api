"""
Integration tests — requires both the FastAPI server and Ollama to be running.

Start the server first:
    uvicorn app.main:app --reload

Then run:
    pytest tests/
"""

import pytest
import httpx

BASE_URL = "http://localhost:8000"
TIMEOUT = 60.0


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
            "POST",
            f"{BASE_URL}/generate",
            json={"prompt": "Say hello in one word."},
        ) as r:
            assert r.status_code == 200
            tokens = [
                line[6:]
                async for line in r.aiter_lines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
    assert len(tokens) > 0


@pytest.mark.asyncio
async def test_summarize_streams_tokens():
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        async with client.stream(
            "POST",
            f"{BASE_URL}/summarize",
            json={
                "text": (
                    "FastAPI is a modern Python web framework for building APIs. "
                    "It is fast, easy to use, and generates automatic interactive docs."
                )
            },
        ) as r:
            assert r.status_code == 200
            tokens = [
                line[6:]
                async for line in r.aiter_lines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
    assert len(tokens) > 0


@pytest.mark.asyncio
async def test_classify_streams_tokens():
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        async with client.stream(
            "POST",
            f"{BASE_URL}/classify",
            json={
                "text": "The stock market crashed today.",
                "categories": ["finance", "sports", "politics"],
            },
        ) as r:
            assert r.status_code == 200
            tokens = [
                line[6:]
                async for line in r.aiter_lines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
    assert len(tokens) > 0


@pytest.mark.asyncio
async def test_extract_keywords_streams_tokens():
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        async with client.stream(
            "POST",
            f"{BASE_URL}/extract-keywords",
            json={
                "text": "Machine learning models require large datasets and significant compute resources."
            },
        ) as r:
            assert r.status_code == 200
            tokens = [
                line[6:]
                async for line in r.aiter_lines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
    assert len(tokens) > 0
