import logging
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from httpx import ConnectError, RemoteProtocolError, TimeoutException

from app import services
from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL
from app.ollama_client import OllamaClient
from app.schemas import (
    ClassifyRequest,
    DescribeImageRequest,
    ExtractKeywordsRequest,
    GenerateRequest,
    SummarizeRequest,
)

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/api.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Local LLM API Playground",
    description="FastAPI wrapper for a locally running LLM via Ollama.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ollama = OllamaClient(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    logger.info(
        "%s %s | %s | %.2fs",
        request.method,
        request.url.path,
        response.status_code,
        duration,
    )
    return response


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


@app.exception_handler(RemoteProtocolError)
async def remote_protocol_error_handler(request, exc):
    return JSONResponse(
        status_code=502,
        content={"error": "Ollama closed the connection unexpectedly. The model may have rejected the input."},
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
        services.generate(
            ollama, request.prompt, request.model,
            request.images, request.temperature, request.max_tokens,
        ),
        media_type="text/event-stream",
    )


@app.post("/describe-image")
async def describe_image(request: DescribeImageRequest):
    return StreamingResponse(
        services.describe_image(
            ollama, request.image, request.prompt,
            request.model, request.temperature, request.max_tokens,
        ),
        media_type="text/event-stream",
    )


@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    return StreamingResponse(
        services.summarize(
            ollama, request.text, request.model,
            request.temperature, request.max_tokens,
        ),
        media_type="text/event-stream",
    )


@app.post("/classify")
async def classify(request: ClassifyRequest):
    return StreamingResponse(
        services.classify(
            ollama, request.text, request.categories,
            request.model, request.temperature, request.max_tokens,
        ),
        media_type="text/event-stream",
    )


@app.post("/extract-keywords")
async def extract_keywords(request: ExtractKeywordsRequest):
    return StreamingResponse(
        services.extract_keywords(
            ollama, request.text, request.model,
            request.temperature, request.max_tokens,
        ),
        media_type="text/event-stream",
    )
