from pydantic import BaseModel


class GenerateRequest(BaseModel):
    prompt: str
    images: list[str] | None = None  # base64-encoded images, optional
    model: str | None = None


class DescribeImageRequest(BaseModel):
    image: str  # base64-encoded
    prompt: str = "Describe this image in detail."
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
