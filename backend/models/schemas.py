from datetime import datetime
from typing import List

from pydantic import BaseModel, ConfigDict


class ChatCreate(BaseModel):
    title: str


class ChatResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    created_at: datetime
    updated_at: datetime


class MessageCreate(BaseModel):
    content: str
    role: str


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    chat_id: int
    content: str
    role: str
    created_at: datetime


class DocumentUploadResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    uploaded_at: datetime
    processed: bool
    num_chunks: int
    query_enabled: bool
    collection_name: str


class DocumentPreferenceUpdate(BaseModel):
    query_enabled: bool


class QueryRequest(BaseModel):
    chat_id: int
    query: str


class SourceDetail(BaseModel):
    label: str
    content: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDetail]
    message_id: int
