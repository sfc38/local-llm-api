from pydantic import BaseModel


class GenerateRequest(BaseModel):
    prompt: str
    model: str | None = None


class SummarizeRequest(BaseModel):
    text: str
    model: str | None = None


class ClassifyRequest(BaseModel):
    text: str
    categories: list[str]
    model: str | None = None


class ExtractKeywordsRequest(BaseModel):
    text: str
    model: str | None = None
