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


# Appointment Scheduling Models

class AssociateInfo(BaseModel):
    """Model for associate information."""
    id: str
    name: str
    email: str
    phone: Optional[str] = None
    specialization: str = "Real Estate"
    availability_hours: str = "9:00 AM - 6:00 PM"
    timezone: str = "America/New_York"


class AvailabilitySlot(BaseModel):
    """Model for availability time slots."""
    datetime: datetime
    associate_id: str
    duration_minutes: int = 60
    is_available: bool = True


class AppointmentRequest(BaseModel):
    """Request model for scheduling an appointment."""
    session_id: str = Field(..., description="Session ID for conversation tracking")
    associate_id: str = Field(..., description="ID of the associate to schedule with")
    user_name: str = Field(..., description="Name of the user scheduling the appointment")
    user_email: str = Field(..., description="Email of the user")
    user_phone: Optional[str] = Field(None, description="Phone number of the user")
    scheduled_time: datetime = Field(..., description="Requested appointment time")
    appointment_type: str = Field(default="consultation", description="Type of appointment")
    notes: Optional[str] = Field(None, description="Additional notes for the appointment")


class AppointmentResponse(BaseModel):
    """Response model for appointment scheduling."""
    success: bool
    appointment_id: Optional[str] = None
    message: str
    scheduled_time: Optional[datetime] = None
    associate_name: Optional[str] = None
    calendar_event_id: Optional[str] = None


class AppointmentDetails(BaseModel):
    """Model for detailed appointment information."""
    id: str
    session_id: str
    associate_id: str
    associate_name: str
    user_name: str
    user_email: str
    user_phone: Optional[str] = None
    scheduled_time: datetime
    appointment_type: str
    status: str = "scheduled"  # scheduled, confirmed, cancelled, completed
    notes: Optional[str] = None
    calendar_event_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class SchedulingOfferRequest(BaseModel):
    """Request model for generating scheduling offers."""
    session_id: str = Field(..., description="Session ID for conversation tracking")
    context: str = Field(default="", description="Context about the current conversation")
    user_message: str = Field(..., description="User's current message")


class SchedulingOfferResponse(BaseModel):
    """Response model for scheduling offers."""
    should_offer: bool
    offer_message: Optional[str] = None
    available_associates: List[AssociateInfo] = Field(default_factory=list)
    next_available_slots: List[AvailabilitySlot] = Field(default_factory=list)


class FollowUpRequest(BaseModel):
    """Request model for follow-up scheduling."""
    session_id: str = Field(..., description="Session ID for conversation tracking")
    days_ahead: int = Field(default=14, description="Number of days ahead to schedule follow-up")
    follow_up_type: str = Field(default="property_search", description="Type of follow-up")


class FollowUpResponse(BaseModel):
    """Response model for follow-up scheduling."""
    success: bool
    message: str
    suggested_time: Optional[datetime] = None
    associate_name: Optional[str] = None


class CalendarIntegrationRequest(BaseModel):
    """Request model for calendar integration."""
    appointment_id: str = Field(..., description="ID of the appointment to add to calendar")
    calendar_type: str = Field(default="google", description="Type of calendar (google, outlook)")


class CalendarIntegrationResponse(BaseModel):
    """Response model for calendar integration."""
    success: bool
    calendar_event_id: Optional[str] = None
    calendar_url: Optional[str] = None
    message: str


class AppointmentListRequest(BaseModel):
    """Request model for listing appointments."""
    session_id: Optional[str] = None
    associate_id: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    status: Optional[str] = None


class AppointmentListResponse(BaseModel):
    """Response model for listing appointments."""
    appointments: List[AppointmentDetails]
    total_count: int
    filtered_count: int 

# Trending Properties

class PropertyEngageRequest(BaseModel):
    property_id: int = Field(..., description="unique_id of property engaged with")
    event_type: str = Field(..., description="view | click | mention")
    session_id: str = Field(..., description="Session ID")

class TrendingProperty(BaseModel):
    property_id: int
    address: str
    score: int

class TrendingResponse(BaseModel):
    trending: List[TrendingProperty] 

class DocumentCountResponse(BaseModel):
    count: int 