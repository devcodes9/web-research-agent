from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List

class ResearchRequest(BaseModel):
    query: str = Field(..., description="The user’s search query")

class Document(BaseModel):
    content: str
    score: float
    # not decided yet

class ResearchResponse(BaseModel):
    query: str
    result: Optional[Document]
