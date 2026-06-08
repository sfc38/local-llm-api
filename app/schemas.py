from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str
    images: list[str] | None = None
    model: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=8192)


class DescribeImageRequest(BaseModel):
    image: str
    prompt: str = "Describe this image in detail."
    model: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=8192)


class SummarizeRequest(BaseModel):
    text: str
    model: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=8192)


class ClassifyRequest(BaseModel):
    text: str
    categories: list[str]
    model: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=8192)


class ExtractKeywordsRequest(BaseModel):
    text: str
    model: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=8192)
