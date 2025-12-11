from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List


class ChatCreate(BaseModel):
    title: str


class ChatResponse(BaseModel):
    id: int
    title: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class MessageCreate(BaseModel):
    content: str
    role: str


class MessageResponse(BaseModel):
    id: int
    chat_id: int
    content: str
    role: str
    created_at: datetime
    
    class Config:
        from_attributes = True


class DocumentUploadResponse(BaseModel):
    id: int
    filename: str
    uploaded_at: datetime
    processed: bool
    num_chunks: int
    
    class Config:
        from_attributes = True


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
