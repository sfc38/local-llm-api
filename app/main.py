from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from httpx import ConnectError, TimeoutException

from app import services
from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL
from app.ollama_client import OllamaClient
from app.schemas import (
    ClassifyRequest,
    ExtractKeywordsRequest,
    GenerateRequest,
    SummarizeRequest,
)

app = FastAPI(
    title="Local LLM API Playground",
    description="FastAPI wrapper for a locally running LLM via Ollama.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ollama = OllamaClient(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)


@app.exception_handler(ConnectError)
async def connect_error_handler(request, exc):
    return JSONResponse(
        status_code=503,
        content={"error": "Ollama is not reachable. Make sure Ollama is running."},
    )


@app.exception_handler(TimeoutException)
async def timeout_error_handler(request, exc):
    return JSONResponse(
        status_code=504,
        content={"error": "Ollama timed out. The model may still be loading."},
    )


@app.get("/health")
async def health():
    if not await ollama.is_reachable():
        raise HTTPException(
            status_code=503,
            detail="Ollama is not reachable. Make sure Ollama is running.",
        )
    return {"status": "ok", "ollama": "reachable", "model": OLLAMA_MODEL}


@app.post("/generate")
async def generate(request: GenerateRequest):
    return StreamingResponse(
        services.generate(ollama, request.prompt, request.model),
        media_type="text/event-stream",
    )


@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    return StreamingResponse(
        services.summarize(ollama, request.text, request.model),
        media_type="text/event-stream",
    )


@app.post("/classify")
async def classify(request: ClassifyRequest):
    return StreamingResponse(
        services.classify(ollama, request.text, request.categories, request.model),
        media_type="text/event-stream",
    )


@app.post("/extract-keywords")
async def extract_keywords(request: ExtractKeywordsRequest):
    return StreamingResponse(
        services.extract_keywords(ollama, request.text, request.model),
        media_type="text/event-stream",
    )
