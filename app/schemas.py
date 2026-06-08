from pydantic import BaseModel, Field


class LLMBase(BaseModel):
    model: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=8192)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1, le=100)
    repeat_penalty: float | None = Field(default=None, ge=0.5, le=2.0)
    seed: int | None = None
    num_ctx: int | None = Field(default=None, ge=512, le=32768)


class GenerateRequest(LLMBase):
    prompt: str
    images: list[str] | None = None


class DescribeImageRequest(LLMBase):
    image: str
    prompt: str = "Describe this image in detail."


class SummarizeRequest(LLMBase):
    text: str


class ClassifyRequest(LLMBase):
    text: str
    categories: list[str]


class ExtractKeywordsRequest(LLMBase):
    text: str


class Message(BaseModel):
    role: str
    content: str
    images: list[str] | None = None


class ChatRequest(LLMBase):
    messages: list[Message]
