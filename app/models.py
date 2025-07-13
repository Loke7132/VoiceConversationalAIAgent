from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import base64


class ConversationMessage(BaseModel):
    """Model for conversation messages."""
    role: str
    content: str
    timestamp: datetime


class TranscribeResponse(BaseModel):
    """Response model for transcription endpoint."""
    text: str
    processing_time: float


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    session_id: str = Field(..., description="Session ID for conversation tracking")
    message: str = Field(..., description="User message to send to the LLM")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str
    session_id: str
    processing_time: float
    metrics: Dict[str, Any] = Field(default_factory=dict)


class SpeakRequest(BaseModel):
    """Request model for text-to-speech endpoint."""
    text: str = Field(..., description="Text to convert to speech")


class SpeakResponse(BaseModel):
    """Response model for text-to-speech endpoint."""
    audio_data: str = Field(..., description="Base64 encoded audio data")
    processing_time: float

    @classmethod
    def from_audio_bytes(cls, audio_bytes: bytes, processing_time: float):
        """Create SpeakResponse from audio bytes."""
        audio_data = base64.b64encode(audio_bytes).decode('utf-8')
        return cls(audio_data=audio_data, processing_time=processing_time)


class ConverseRequest(BaseModel):
    """Request model for end-to-end conversation endpoint."""
    session_id: str = Field(..., description="Session ID for conversation tracking")


class ReactRequest(BaseModel):
    """Request model for ReAct conversation endpoint."""
    session_id: str = Field(..., description="Session ID for conversation tracking")
    message: str = Field(..., description="User message to process with ReAct")


class ReactResponse(BaseModel):
    """Response model for ReAct conversation endpoint."""
    transcription: str
    llm_response: str
    session_id: str
    processing_time: float
    metrics: Dict[str, Any] = Field(default_factory=dict)


class ConverseResponse(BaseModel):
    """Response model for end-to-end conversation endpoint."""
    transcription: str
    llm_response: str
    audio_response: str = Field(..., description="Base64 encoded audio response")
    session_id: str
    processing_time: float
    metrics: Dict[str, Any] = Field(default_factory=dict)


class ResetResponse(BaseModel):
    """Response model for reset endpoint."""
    success: bool
    session_id: str
    message: str


class UploadResult(BaseModel):
    """Result model for individual file upload."""
    filename: str
    success: bool
    error: Optional[str] = None
    processing_time: Optional[float] = None


class UploadResponse(BaseModel):
    """Response model for document upload endpoint."""
    results: List[UploadResult]
    total_processing_time: float


class DocumentMetadata(BaseModel):
    """Model for document metadata."""
    id: str
    filename: str
    file_type: str
    upload_timestamp: datetime
    processed_status: str
    chunk_count: Optional[int] = None


class DocumentChunk(BaseModel):
    """Model for document chunks."""
    id: str
    document_id: str
    chunk_text: str
    chunk_order: int
    embedding: Optional[List[float]] = None


class RAGContext(BaseModel):
    """Model for RAG context retrieval."""
    chunks: List[DocumentChunk]
    similarity_scores: List[float]
    context_text: str


class ErrorResponse(BaseModel):
    """Model for error responses."""
    detail: str
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class HealthResponse(BaseModel):
    """Model for health check response."""
    status: str
    timestamp: datetime = Field(default_factory=datetime.now)
    services: Dict[str, str] = Field(default_factory=dict)


class MetricsResponse(BaseModel):
    """Model for performance metrics response."""
    total_requests: int
    average_response_time: float
    success_rate: float
    error_count: int
    last_updated: datetime = Field(default_factory=datetime.now) 