from pydantic import BaseModel, Field
from typing import List, Optional

VALID_ROLES = {"system", "user", "assistant"}

class ChatMessage(BaseModel):
    role: str = Field(pattern=r"^(system|user|assistant)$", description="Message role")
    content: str = Field(min_length=1, max_length=4096, description="Message content")

class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(min_length=1, max_length=50)

class ChatResponse(BaseModel):
    content: str

class ExplainRepoRequest(BaseModel):
    full_name: str = Field(min_length=1, max_length=200, description="GitHub repo full name (owner/repo)")
    question: str = Field(default="", max_length=500, description="Optional specific question about the repo")
