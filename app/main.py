from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import time
from datetime import datetime, timedelta
import asyncio
import io
import base64
import re
import csv
from pathlib import Path
from fastapi import Depends

from .config import settings
from .models import (
    TranscribeResponse, 
    ChatRequest, 
    ChatResponse, 
    SpeakRequest, 
    SpeakResponse,
    ConverseRequest,
    ConverseResponse,
    ReactRequest,
    ReactResponse,
    ResetResponse,
    UploadResponse,
    ConversationMessage,
    # Appointment models
    AppointmentRequest,
    AppointmentResponse,
    SchedulingOfferRequest,
    SchedulingOfferResponse,
    FollowUpRequest,
    FollowUpResponse,
    AssociateInfo,
    AvailabilitySlot,
    AppointmentDetails,
    AppointmentListRequest,
    AppointmentListResponse,
    PropertyEngageRequest,
    TrendingResponse,
    DocumentCountResponse
)
from .services.supabase_service import SupabaseService
from .services.gemini_service import GeminiService
from .services.google_cloud_service import GoogleCloudService
from .services.rag_service import RAGService
from .services.translation_service import TranslationService
from .utils.performance import PerformanceTracker
from .services.react_service import ReActService
from .services.appointment_service import AppointmentService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_llm_response(response: str) -> str:
    """Clean LLM response by removing formatting markers and unwanted characters."""
    if not response:
        return response
    
    # Remove formatting markers
    clean_response = response.replace('**', '').replace('*', '')
    clean_response = clean_response.replace('###', '').replace('#', '')
    clean_response = clean_response.replace('||', '').replace('|', '')
    clean_response = clean_response.replace('--', '-').replace('- -', '-')
    
    # Remove extra whitespace but preserve paragraph breaks
    clean_response = re.sub(r'[ \t]+', ' ', clean_response)  # Multiple spaces/tabs to single space
    clean_response = re.sub(r'\n[ \t]+', '\n', clean_response)  # Remove leading whitespace on lines
    clean_response = re.sub(r'[ \t]+\n', '\n', clean_response)  # Remove trailing whitespace on lines
    clean_response = re.sub(r'\n{3,}', '\n\n', clean_response)  # Limit to max 2 consecutive newlines
    
    return clean_response.strip()

# Initialize FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="Voice Conversational Agentic AI with RAG capabilities",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
supabase_service = SupabaseService()
gemini_service = GeminiService()
speech_service = GoogleCloudService()
translation_service = TranslationService()
rag_service = RAGService(supabase_service)
react_service = ReActService(rag_service, supabase_service, gemini_service)
appointment_service = AppointmentService(supabase_service)
class AppointmentMessageRequest(BaseModel):
    appointment_id: str
    message: str
    user_type: str = 'associate'

class AppointmentMessageResponse(BaseModel):
    id: str
    sent_at: datetime

# ---------------- Property Data -----------------

class PropertyRecord(BaseModel):
    unique_id: int
    address: str
    floor: str | None = None
    suite: str | None = None
    size_sf: int | None = None
    rent_per_sf_year: str | None = None


_PROPERTIES_CACHE: List[PropertyRecord] | None = None


def _load_properties() -> List[PropertyRecord]:
    global _PROPERTIES_CACHE
    if _PROPERTIES_CACHE is not None:
        return _PROPERTIES_CACHE

    csv_path = Path(__file__).parent.parent / "HackathonInternalKnowledgeBase.csv"
    if not csv_path.exists():
        logger.warning("Property CSV not found: %s", csv_path)
        _PROPERTIES_CACHE = []
        return _PROPERTIES_CACHE

    props: List[PropertyRecord] = []
    with csv_path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                props.append(
                    PropertyRecord(
                        unique_id=int(row.get("unique_id", len(props)+1)),
                        address=row.get("Property Address") or "",
                        floor=row.get("Floor"),
                        suite=row.get("Suite"),
                        size_sf=int(str(row.get("Size (SF)", "0")).replace(",", "")) if row.get("Size (SF)") else None,
                        rent_per_sf_year=row.get("Rent/SF/Year")
                    )
                )
            except Exception as e:
                logger.warning("Failed parsing property row: %s", e)
    _PROPERTIES_CACHE = props
    return props


@app.get("/properties", response_model=List[PropertyRecord])
async def get_properties():
    """Return list of properties from the knowledge base."""
    return _load_properties()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting Voice Conversational Agentic AI API...")
    await rag_service.initialize_embedding_model()
    logger.info("API startup complete.")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Voice Conversational Agentic AI API...")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Voice Conversational Agentic AI API",
        "version": settings.api_version,
        "endpoints": {
            "transcribe": "/transcribe",
            "chat": "/chat",
            "speak": "/speak",
            "converse": "/converse",
            "converse_multilingual": "/converse_multilingual",
            "converse_react": "/converse_react",
            "converse_react_voice": "/converse_react_voice",
            "supported_languages": "/supported_languages",
            "reset": "/reset",
            "upload_rag_docs": "/upload_rag_docs",
            # Appointment scheduling endpoints
            "schedule_appointment": "/schedule_appointment",
            "associates": "/associates",
            "associate_availability": "/associates/{associate_id}/availability",
            "scheduling_offer": "/scheduling_offer",
            "follow_up": "/follow_up",
            "appointment_details": "/appointments/{appointment_id}",
            "cancel_appointment": "/appointments/{appointment_id}/cancel",
            "reschedule_appointment": "/appointments/{appointment_id}/reschedule",
            "session_appointments": "/appointments/session/{session_id}"
        }
    }

@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(
    audio: UploadFile = File(..., description="Audio file to transcribe")
):
    """
    Convert speech to text using Eleven Labs STT.
    
    Args:
        audio: Audio file (WAV, MP3, etc.)
        
    Returns:
        TranscribeResponse: Transcription text and processing time
    """
    tracker = PerformanceTracker()
    tracker.start("transcribe")
    
    try:
        # Read audio file
        audio_data = await audio.read()
        
        # Call Google Cloud STT service
        transcription = await speech_service.transcribe_audio(audio_data)
        
        tracker.end("transcribe")
        
        return TranscribeResponse(
            text=transcription,
            processing_time=tracker.get_duration("transcribe")
        )
        
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat_with_llm(request: ChatRequest):
    """
    Chat with the LLM using RAG context and conversation history.
    
    Args:
        request: Chat request with session_id and message
        
    Returns:
        ChatResponse: LLM response with processing metrics
    """
    tracker = PerformanceTracker()
    tracker.start("chat_total")
    
    try:
        # Load conversation history
        tracker.start("load_history")
        conversation_history = await supabase_service.get_conversation_history(request.session_id)
        tracker.end("load_history")
        
        # Get RAG context
        tracker.start("rag_retrieval")
        
        # Enhanced query processing for contextual references
        enhanced_query = request.message
        is_contextual_query = False
        
        # If the query contains contextual references, enhance it with conversation history
        contextual_references = ['that address', 'this property', 'the same location', 'that property', 'this address', 'the property', 'this', 'more details', 'details about', 'tell me more', 'about this']
        if any(ref in request.message.lower() for ref in contextual_references):
            is_contextual_query = True
            # Extract property addresses from recent conversation history
            recent_history = conversation_history[-5:] if len(conversation_history) > 5 else conversation_history
            for msg in reversed(recent_history):
                if msg.role == 'assistant':
                    # Look for property addresses in assistant responses
                    import re
                    address_patterns = [
                        r'\b\d+\s+[NSEW]?\s*\w+\s+\d+\w+\s+(?:St|Street)\b',
                        r'\b\d+\s+\w+\s+(?:Avenue|Ave)\b',
                        r'\b\d+\s+\w+\s+(?:Square|Sq|Plaza|Blvd|Boulevard|Way|Road|Rd)\b',
                        r'Address[:\s]+([^,\n]+)',
                        r'Property[:\s]+([^,\n]+)',
                        r'(\d+\s+\w+\s+(?:Square|Sq|Plaza),?\s+Floor\s+[A-Z]?\d+,?\s+Suite\s+\d+[A-Z]?)',
                        r'(\d+\s+[NSEW]?\s*\w+\s+\d+\w+\s+(?:St|Street),?\s+Floor\s+[A-Z]?\d+,?\s+Suite\s+\d+[A-Z]?)',
                        r'(\d+\s+\w+\s+(?:Avenue|Ave),?\s+Floor\s+[A-Z]?\d+,?\s+Suite\s+\d+[A-Z]?)'
                    ]
                    
                    for pattern in address_patterns:
                        matches = re.findall(pattern, msg.content, re.IGNORECASE)
                        if matches:
                            # Use the first found address to enhance the query
                            found_address = matches[0]
                            enhanced_query = f"{request.message} {found_address}"
                            logger.info(f"Enhanced contextual query: {enhanced_query}")
                            break
                    
                    if enhanced_query != request.message:
                        break
        
        # Use comprehensive data retrieval for contextual queries to maintain consistency with ReAct
        if is_contextual_query:
            logger.info("Using comprehensive data retrieval for contextual query")
            rag_context = await rag_service.get_relevant_context(
                enhanced_query, 
                max_chunks=250,
                similarity_threshold=0.001  # Low threshold for address matching
            )
        else:
            rag_context = await rag_service.get_relevant_context(
                enhanced_query, 
                similarity_threshold=0.001  # Low threshold for address matching
            )
        tracker.end("rag_retrieval")
        
        # Build messages for LLM
        messages = []
        
        # Add system prompt with RAG context
        if rag_context:
            system_prompt = f"""You are a helpful real estate AI assistant with access to a property database. When users ask about properties, use the following property data to provide accurate, specific information.

Property Data:
{rag_context}

CRITICAL CONTEXT UNDERSTANDING:
1. For queries with contextual references (like "that address", "this property", "the same location"):
   - Review the conversation history to identify the specific property being referenced
   - Extract the exact address, floor, and suite from previous responses
   - Search for and provide information about that specific property
   - If you cannot find the referenced property in the current data, explicitly state which property was referenced and that you need more specific information

2. For follow-up questions about previously mentioned properties:
   - Always check if the current property data contains the referenced property
   - If the property is not in the current data but was mentioned before, explain this clearly
   - Provide helpful guidance on how to get the needed information

CRITICAL NUMERICAL COMPARISON REQUIREMENTS:
1. For superlative queries (maximum, minimum, highest, lowest, largest, smallest):
   - ALWAYS perform numerical comparison on ALL provided data above
   - Extract and compare actual numerical values (rent amounts, sizes, rates)
   - Do NOT rely on text descriptions - calculate the actual maximum/minimum from the data provided
   - For "maximum monthly rent" - compare ALL monthly rent values in the data and find the highest
   - For "largest size" - compare ALL size values in the data and find the biggest
   - For "highest rate" - compare ALL rate values in the data and find the maximum
   - IMPORTANT: Work with the data provided above - find the best answer from the data provided

2. For comparison queries (greater than, less than, between):
   - Parse the numerical criteria from the query
   - Filter and compare actual numerical values from the provided data
   - Show only properties that meet the numerical criteria

3. For range queries:
   - Extract the range boundaries from the query
   - Compare each property's values against the range from the provided data
   - Include properties that fall within the specified range

4. Data Limitations Transparency:
   - If you find a maximum/minimum from the provided data, present it as the answer
   - You are working with a subset of the database - find the best answer from the data provided
   - Focus on providing accurate analysis of the data you have access to

CRITICAL FORMATTING REQUIREMENTS - ALWAYS FOLLOW:
1. NEVER use paragraph format - ALWAYS use bullet points or numbered lists
2. For single property queries, use bullet points for details
3. For multiple properties, use numbered lists
4. Each property should be on a separate line with clear formatting
5. Always include key details: address, floor, suite, size, monthly rent, rate per sq ft

RESPONSE FORMAT TEMPLATES:

For Single Property:
• **Address**: [Property Address], Floor [Floor], Suite [Suite]
• **Size**: [Size] square feet
• **Monthly Rent**: $[Amount]
• **Rate**: $[Rate] per square foot annually
• **Associates**: [List associates if requested]
• **GCI On 3 Years**: $[Amount]
• **Additional Info**: [Any other requested details]

IMPORTANT: Never include Unique ID or internal database identifiers in responses to users.

For Multiple Properties:
1. **[Address]** - Floor [Floor], Suite [Suite]
   • Size: [Size] sq ft
   • Monthly Rent: $[Amount]
   • Rate: $[Rate]/sq ft annually

2. **[Address]** - Floor [Floor], Suite [Suite]
   • Size: [Size] sq ft
   • Monthly Rent: $[Amount]
   • Rate: $[Rate]/sq ft annually

For Comparisons/Lists:
• **Property with Highest Monthly Rent**: [Address] with $[Amount]/month
• **Largest Property**: [Address] with [Size] sq ft
• **Best Rate**: [Address] at $[Rate]/sq ft annually

For Column-Specific Queries:
• **Associate 1**: [Name]
• **Associate 2**: [Name]
• **Associate 3**: [Name]
• **Broker Email**: [Email]
• **GCI On 3 Years**: $[Amount]

CRITICAL: Do NOT include Unique ID, internal database IDs, or system identifiers in any user-facing responses.

NUMERICAL ANALYSIS EXAMPLES:
- "maximum monthly rent" → Find the property with the highest monthly rent value from the data provided
- "largest property" → Find the property with the highest size value from the data provided
- "rent greater than $150,000" → Filter properties where monthly rent > $150,000 from the data provided
- "size between 15000 and 20000" → Filter properties where size is 15000-20000 sq ft from the data provided

RULES:
- Use bullet points (•) for details within each property
- Use numbers (1, 2, 3) for listing multiple properties
- Use bold (**text**) for field labels and property addresses
- Never write long paragraphs - break everything into clear points
- Always start responses with a brief intro line, then use structured format
- For size comparisons, clearly state "closest matches" or "no exact match found"
- ALWAYS perform actual numerical calculations for comparisons on the provided data
- When multiple properties meet criteria, list them in order (highest to lowest, etc.)
- Work with the data provided above to find the best answers

APPOINTMENT BOOKING INSTRUCTIONS:
When users ask to book or schedule appointments:
- DO NOT say you cannot book appointments directly
- DO NOT provide broker contact information for booking
- Instead, provide property information and let the system handle the booking flow
- If users ask about booking, focus on providing property details first
- The system will automatically detect booking requests and offer scheduling

Always provide helpful, structured responses using the property data above with accurate numerical analysis."""
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        for msg in conversation_history:
            messages.append({"role": msg.role, "content": msg.content})
        
        # Add current user message
        messages.append({"role": "user", "content": request.message})
        
        # Handle appointment flow first
        tracker.start("appointment_flow")
        try:
            appointment_response = await appointment_service.handle_appointment_flow(
                request.message, 
                request.session_id, 
                [{"role": msg.role, "content": msg.content} for msg in conversation_history]
            )
            
            if appointment_response:
                # Appointment was handled, use the appointment response
                llm_response = appointment_response
                logger.info(f"Appointment automatically handled for session {request.session_id}")
            else:
                # No appointment handling, proceed with normal LLM response
                tracker.start("llm_processing")
                llm_response = await gemini_service.generate_response(messages)
                # Clean the response to remove formatting markers
                llm_response = clean_llm_response(llm_response)
                tracker.end("llm_processing")
                
                # Check if we should offer appointment scheduling
                tracker.start("appointment_detection")
                try:
                    should_offer_scheduling = appointment_service.should_offer_scheduling(
                        request.message, 
                        [{"role": msg.role, "content": msg.content} for msg in conversation_history]
                    )
                    
                    logger.info(f"Appointment detection - Message: '{request.message}', Should offer: {should_offer_scheduling}")
                    
                    if should_offer_scheduling:
                        # Generate scheduling offer
                        try:
                            scheduling_offer = appointment_service.generate_scheduling_offer(llm_response)
                            
                            # Append scheduling offer to LLM response
                            llm_response += f"\n\n{scheduling_offer}"
                            
                            logger.info(f"Added scheduling offer to response for session {request.session_id}: {scheduling_offer}")
                        except Exception as e:
                            logger.error(f"Error generating scheduling offer: {str(e)}")
                            # Add a simple fallback offer
                            fallback_offer = "Would you like to schedule a meeting with one of our associates? I can help you book an appointment."
                            llm_response += f"\n\n{fallback_offer}"
                            logger.info(f"Added fallback scheduling offer for session {request.session_id}")
                except Exception as e:
                    logger.error(f"Error in appointment detection: {str(e)}")
                tracker.end("appointment_detection")
        except Exception as e:
            logger.error(f"Error in appointment flow: {str(e)}")
            # Fallback to normal LLM processing
            tracker.start("llm_processing")
            llm_response = await gemini_service.generate_response(messages)
            llm_response = clean_llm_response(llm_response)
            tracker.end("llm_processing")
        tracker.end("appointment_flow")
        
        # Save conversation to database
        tracker.start("save_conversation")
        await supabase_service.save_conversation_message(
            request.session_id, "user", request.message
        )
        await supabase_service.save_conversation_message(
            request.session_id, "assistant", llm_response
        )
        tracker.end("save_conversation")
        
        tracker.end("chat_total")
        
        return ChatResponse(
            response=llm_response,
            session_id=request.session_id,
            processing_time=tracker.get_duration("chat_total"),
            metrics={
                "load_history_time": tracker.get_duration("load_history"),
                "rag_retrieval_time": tracker.get_duration("rag_retrieval"),
                "llm_processing_time": tracker.get_duration("llm_processing"),
                "appointment_detection_time": tracker.get_duration("appointment_detection"),
                "save_conversation_time": tracker.get_duration("save_conversation")
            }
        )
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.post("/speak", response_model=SpeakResponse)
async def text_to_speech(request: SpeakRequest):
    """
    Convert text to speech using Eleven Labs TTS.
    
    Args:
        request: Text to convert to speech
        
    Returns:
        SpeakResponse: Audio data and processing time
    """
    tracker = PerformanceTracker()
    tracker.start("tts")
    
    try:
        # Call Google Cloud TTS service
        audio_bytes = await speech_service.text_to_speech(request.text)
        
        tracker.end("tts")
        
        return SpeakResponse.from_audio_bytes(
            audio_bytes=audio_bytes,
            processing_time=tracker.get_duration("tts")
        )
        
    except Exception as e:
        logger.error(f"TTS error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")

@app.post("/converse", response_model=ConverseResponse)
async def end_to_end_conversation(
    audio: UploadFile = File(..., description="Audio file with user's voice"),
    session_id: str = Form(..., description="Session ID for conversation tracking")
):
    """
    End-to-end conversation: STT -> LLM with RAG -> TTS.
    
    Args:
        audio: Audio file with user's voice
        session_id: Session ID for conversation tracking
        
    Returns:
        ConverseResponse: Complete conversation result with all metrics
    """
    tracker = PerformanceTracker()
    tracker.start("converse_total")
    
    try:
        # Step 1: Speech to Text
        tracker.start("stt")
        audio_data = await audio.read()
        transcription = await speech_service.transcribe_audio(audio_data)
        tracker.end("stt")
        
        # Step 2: Chat with LLM (includes RAG)
        tracker.start("chat")
        chat_request = ChatRequest(session_id=session_id, message=transcription)
        
        # Load conversation history
        conversation_history = await supabase_service.get_conversation_history(session_id)
        
        # Check if this is a superlative query that would benefit from ReAct
        is_superlative = any(word in transcription.lower() for word in ['maximum', 'minimum', 'highest', 'lowest', 'largest', 'smallest', 'most', 'least', 'max', 'min'])
        
        if is_superlative:
            logger.info(f"Detected superlative query, using ReAct: {transcription}")
            try:
                final_answer, react_steps = await react_service.process_query(transcription, session_id)
                tracker.end("chat")
                
                # Step 3: Text to Speech
                tracker.start("tts")
                audio_bytes = await speech_service.text_to_speech(final_answer)
                # Convert bytes to base64-encoded string
                audio_response = base64.b64encode(audio_bytes).decode('utf-8')
                tracker.end("tts")
                
                tracker.end("converse_total")
                
                return ConverseResponse(
                    transcription=transcription,
                    llm_response=final_answer,
                    audio_response=audio_response,
                    session_id=session_id,
                    processing_time=tracker.get_duration("converse_total"),
                    metrics={
                        "stt_time": tracker.get_duration("stt"),
                        "chat_time": tracker.get_duration("chat"),
                        "tts_time": tracker.get_duration("tts"),
                        "react_steps": len(react_steps),
                        "processing_method": "ReAct",
                        "context_found": True
                    }
                )
            except Exception as e:
                logger.error(f"ReAct processing failed, falling back to regular RAG: {e}")
        
        # Enhanced query processing for contextual references
        enhanced_query = transcription
        is_contextual_query = False
        
        # If the query contains contextual references, enhance it with conversation history
        contextual_references = ['that address', 'this property', 'the same location', 'that property', 'this address', 'the property', 'this', 'more details', 'details about', 'tell me more', 'about this']
        if any(ref in transcription.lower() for ref in contextual_references):
            is_contextual_query = True
            # Extract property addresses from recent conversation history
            recent_history = conversation_history[-5:] if len(conversation_history) > 5 else conversation_history
            for msg in reversed(recent_history):
                if msg.role == 'assistant':
                    # Look for property addresses in assistant responses
                    import re
                    address_patterns = [
                        r'\b\d+\s+[NSEW]?\s*\w+\s+\d+\w+\s+(?:St|Street)\b',
                        r'\b\d+\s+\w+\s+(?:Avenue|Ave)\b',
                        r'\b\d+\s+\w+\s+(?:Square|Sq|Plaza|Blvd|Boulevard|Way|Road|Rd)\b',
                        r'Address[:\s]+([^,\n]+)',
                        r'Property[:\s]+([^,\n]+)',
                        r'(\d+\s+\w+\s+(?:Square|Sq|Plaza),?\s+Floor\s+[A-Z]?\d+,?\s+Suite\s+\d+[A-Z]?)',
                        r'(\d+\s+[NSEW]?\s*\w+\s+\d+\w+\s+(?:St|Street),?\s+Floor\s+[A-Z]?\d+,?\s+Suite\s+\d+[A-Z]?)',
                        r'(\d+\s+\w+\s+(?:Avenue|Ave),?\s+Floor\s+[A-Z]?\d+,?\s+Suite\s+\d+[A-Z]?)'
                    ]
                    
                    for pattern in address_patterns:
                        matches = re.findall(pattern, msg.content, re.IGNORECASE)
                        if matches:
                            # Use the first found address to enhance the query
                            found_address = matches[0]
                            enhanced_query = f"{transcription} {found_address}"
                            logger.info(f"Enhanced contextual query: {enhanced_query}")
                            break
                    
                    if enhanced_query != transcription:
                        break
        
        # Use comprehensive data retrieval for contextual queries to maintain consistency with ReAct
        if is_contextual_query:
            logger.info("Using comprehensive data retrieval for contextual query")
            rag_context = await rag_service.get_relevant_context(
                enhanced_query, 
                max_chunks=250,
                similarity_threshold=0.001  # Low threshold for address matching
            )
        else:
            rag_context = await rag_service.get_relevant_context(
                enhanced_query, 
                similarity_threshold=0.001  # Low threshold for address matching
            )
        
        # Build messages for LLM
        messages = []
        
        if rag_context:
            system_prompt = f"""You are a helpful real estate AI assistant with access to a property database. When users ask about properties, use the following property data to provide accurate, specific information.

Property Data:
{rag_context}

CRITICAL CONTEXT UNDERSTANDING:
1. For queries with contextual references (like "that address", "this property", "the same location"):
   - Review the conversation history to identify the specific property being referenced
   - Extract the exact address, floor, and suite from previous responses
   - Search for and provide information about that specific property
   - If you cannot find the referenced property in the current data, explicitly state which property was referenced and that you need more specific information

2. For follow-up questions about previously mentioned properties:
   - Always check if the current property data contains the referenced property
   - If the property is not in the current data but was mentioned before, explain this clearly
   - Provide helpful guidance on how to get the needed information

CRITICAL NUMERICAL COMPARISON REQUIREMENTS:
1. For superlative queries (maximum, minimum, highest, lowest, largest, smallest):
   - ALWAYS perform numerical comparison on ALL provided data above
   - Extract and compare actual numerical values (rent amounts, sizes, rates)
   - Do NOT rely on text descriptions - calculate the actual maximum/minimum from the data provided
   - For "maximum monthly rent" - compare ALL monthly rent values in the data and find the highest
   - For "largest size" - compare ALL size values in the data and find the biggest
   - For "highest rate" - compare ALL rate values in the data and find the maximum
   - IMPORTANT: Work with the data provided above - find the best answer from the data provided

2. For comparison queries (greater than, less than, between):
   - Parse the numerical criteria from the query
   - Filter and compare actual numerical values from the provided data
   - Show only properties that meet the numerical criteria

3. For range queries:
   - Extract the range boundaries from the query
   - Compare each property's values against the range from the provided data
   - Include properties that fall within the specified range

4. Data Limitations Transparency:
   - If you find a maximum/minimum from the provided data, present it as the answer
   - You are working with a subset of the database - find the best answer from the data provided
   - Focus on providing accurate analysis of the data you have access to

CRITICAL FORMATTING REQUIREMENTS - ALWAYS FOLLOW:
1. NEVER use paragraph format - ALWAYS use bullet points or numbered lists
2. For single property queries, use bullet points for details
3. For multiple properties, use numbered lists
4. Each property should be on a separate line with clear formatting
5. Always include key details: address, floor, suite, size, monthly rent, rate per sq ft

RESPONSE FORMAT TEMPLATES:

For Single Property:
• **Address**: [Property Address], Floor [Floor], Suite [Suite]
• **Size**: [Size] square feet
• **Monthly Rent**: $[Amount]
• **Rate**: $[Rate] per square foot annually
• **Associates**: [List associates if requested]
• **Additional Info**: [Any other requested details]

IMPORTANT: Never include Unique ID or internal database identifiers in responses to users.

For Multiple Properties:
1. **[Address]** - Floor [Floor], Suite [Suite]
   • Size: [Size] sq ft
   • Monthly Rent: $[Amount]
   • Rate: $[Rate]/sq ft annually

2. **[Address]** - Floor [Floor], Suite [Suite]
   • Size: [Size] sq ft
   • Monthly Rent: $[Amount]
   • Rate: $[Rate]/sq ft annually

For Comparisons/Lists:
• **Property with Highest Monthly Rent**: [Address] with $[Amount]/month
• **Largest Property**: [Address] with [Size] sq ft
• **Best Rate**: [Address] at $[Rate]/sq ft annually

For Column-Specific Queries:
• **Associate 1**: [Name]
• **Associate 2**: [Name]
• **Associate 3**: [Name]
• **Broker Email**: [Email]

CRITICAL: Do NOT include Unique ID, internal database IDs, or system identifiers in any user-facing responses.

NUMERICAL ANALYSIS EXAMPLES:
- "maximum monthly rent" → Find the property with the highest monthly rent value from the data provided
- "largest property" → Find the property with the highest size value from the data provided
- "rent greater than $150,000" → Filter properties where monthly rent > $150,000 from the data provided
- "size between 15000 and 20000" → Filter properties where size is 15000-20000 sq ft from the data provided

RULES:
- Use bullet points (•) for details within each property
- Use numbers (1, 2, 3) for listing multiple properties
- Use bold (**text**) for field labels and property addresses
- Never write long paragraphs - break everything into clear points
- Always start responses with a brief intro line, then use structured format
- For size comparisons, clearly state "closest matches" or "no exact match found"
- ALWAYS perform actual numerical calculations for comparisons on the provided data
- When multiple properties meet criteria, list them in order (highest to lowest, etc.)
- Work with the data provided above to find the best answers

Always provide helpful, structured responses using the property data above with accurate numerical analysis."""
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        for msg in conversation_history:
            messages.append({"role": msg.role, "content": msg.content})
        
        # Add current user message
        messages.append({"role": "user", "content": transcription})
        
        # Get LLM response
        llm_response = await gemini_service.generate_response(messages)
        # Clean the response to remove formatting markers
        llm_response = clean_llm_response(llm_response)
        
        # Save conversation to database
        await supabase_service.save_conversation_message(session_id, "user", transcription)
        await supabase_service.save_conversation_message(session_id, "assistant", llm_response)
        
        tracker.end("chat")
        
        # Step 3: Text to Speech
        tracker.start("tts")
        audio_bytes = await speech_service.text_to_speech(llm_response)
        # Convert bytes to base64-encoded string
        audio_response = base64.b64encode(audio_bytes).decode('utf-8')
        tracker.end("tts")
        
        tracker.end("converse_total")
        
        return ConverseResponse(
            transcription=transcription,
            llm_response=llm_response,
            audio_response=audio_response,
            session_id=session_id,
            processing_time=tracker.get_duration("converse_total"),
            metrics={
                "stt_time": tracker.get_duration("stt"),
                "chat_time": tracker.get_duration("chat"),
                "tts_time": tracker.get_duration("tts")
            }
        )
        
    except Exception as e:
        logger.error(f"Conversation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Conversation failed: {str(e)}")

@app.post("/converse_multilingual", response_model=ConverseResponse)
async def multilingual_conversation(
    audio: UploadFile = File(..., description="Audio file with user's voice"),
    session_id: str = Form(..., description="Session ID for conversation tracking")
):
    """
    End-to-end multilingual conversation: 
    STT with language detection -> Translate to English -> LLM with RAG -> Translate back -> TTS in original language.
    
    Args:
        audio: Audio file with user's voice
        session_id: Session ID for conversation tracking
        
    Returns:
        ConverseResponse: Complete conversation result with all metrics
    """
    tracker = PerformanceTracker()
    tracker.start("converse_multilingual_total")
    
    try:
        # Step 1: Speech to Text with Better Multilingual Support
        tracker.start("stt_multilingual")
        audio_data = await audio.read()
        transcription, detected_language = await speech_service.transcribe_audio_multilingual(audio_data)
        tracker.end("stt_multilingual")
        
        # Convert language code from STT format to ISO 639-1 format for translation
        original_language = await speech_service.convert_language_code("stt", "iso", detected_language)
        
        # Step 2: Translate to English (if not already English)
        tracker.start("translate_to_english")
        english_transcription = transcription
        if original_language != "en":
            english_transcription = await translation_service.translate_text(
                transcription, "en", original_language
            )
        tracker.end("translate_to_english")
        
        # Step 3: Chat with LLM (includes RAG) - using English
        tracker.start("chat")
        chat_request = ChatRequest(session_id=session_id, message=english_transcription)
        
        # Load conversation history
        conversation_history = await supabase_service.get_conversation_history(session_id)
        
        # Enhanced query processing for contextual references
        enhanced_query = english_transcription
        is_contextual_query = False
        
        # If the query contains contextual references, enhance it with conversation history
        contextual_references = ['that address', 'this property', 'the same location', 'that property', 'this address', 'the property', 'this', 'more details', 'details about', 'tell me more', 'about this']
        if any(ref in english_transcription.lower() for ref in contextual_references):
            is_contextual_query = True
            # Extract property addresses from recent conversation history
            recent_history = conversation_history[-5:] if len(conversation_history) > 5 else conversation_history
            for msg in reversed(recent_history):
                if msg.role == 'assistant':
                    # Look for property addresses in assistant responses
                    import re
                    address_patterns = [
                        r'\b\d+\s+[NSEW]?\s*\w+\s+\d+\w+\s+(?:St|Street)\b',
                        r'\b\d+\s+\w+\s+(?:Avenue|Ave)\b',
                        r'\b\d+\s+\w+\s+(?:Square|Sq|Plaza|Blvd|Boulevard|Way|Road|Rd)\b',
                        r'Address[:\s]+([^,\n]+)',
                        r'Property[:\s]+([^,\n]+)',
                        r'(\d+\s+\w+\s+(?:Square|Sq|Plaza),?\s+Floor\s+[A-Z]?\d+,?\s+Suite\s+\d+[A-Z]?)',
                        r'(\d+\s+[NSEW]?\s*\w+\s+\d+\w+\s+(?:St|Street),?\s+Floor\s+[A-Z]?\d+,?\s+Suite\s+\d+[A-Z]?)',
                        r'(\d+\s+\w+\s+(?:Avenue|Ave),?\s+Floor\s+[A-Z]?\d+,?\s+Suite\s+\d+[A-Z]?)'
                    ]
                    
                    for pattern in address_patterns:
                        matches = re.findall(pattern, msg.content, re.IGNORECASE)
                        if matches:
                            # Use the first found address to enhance the query
                            found_address = matches[0]
                            enhanced_query = f"{english_transcription} {found_address}"
                            logger.info(f"Enhanced contextual query: {enhanced_query}")
                            break
                    
                    if enhanced_query != english_transcription:
                        break
        
        # Use comprehensive data retrieval for contextual queries to maintain consistency with ReAct
        if is_contextual_query:
            logger.info("Using comprehensive data retrieval for contextual query")
            rag_context = await rag_service.get_relevant_context(
                enhanced_query, 
                max_chunks=250,
                similarity_threshold=0.001  # Low threshold for address matching
            )
        else:
            rag_context = await rag_service.get_relevant_context(
                enhanced_query, 
                similarity_threshold=0.001  # Low threshold for address matching
            )
        
        # Build messages for LLM
        messages = []
        
        if rag_context:
            system_prompt = f"""You are a helpful real estate AI assistant with access to a property database. When users ask about properties, use the following property data to provide accurate, specific information.

Property Data:
{rag_context}

CRITICAL CONTEXT UNDERSTANDING:
1. For queries with contextual references (like "that address", "this property", "the same location"):
   - Review the conversation history to identify the specific property being referenced
   - Extract the exact address, floor, and suite from previous responses
   - Search for and provide information about that specific property
   - If you cannot find the referenced property in the current data, explicitly state which property was referenced and that you need more specific information

2. For follow-up questions about previously mentioned properties:
   - Always check if the current property data contains the referenced property
   - If the property is not in the current data but was mentioned before, explain this clearly
   - Provide helpful guidance on how to get the needed information

CRITICAL NUMERICAL COMPARISON REQUIREMENTS:
1. For superlative queries (maximum, minimum, highest, lowest, largest, smallest):
   - ALWAYS perform numerical comparison on ALL provided data above
   - Extract and compare actual numerical values (rent amounts, sizes, rates)
   - Do NOT rely on text descriptions - calculate the actual maximum/minimum from the data provided
   - For "maximum monthly rent" - compare ALL monthly rent values in the data and find the highest
   - For "largest size" - compare ALL size values in the data and find the biggest
   - For "highest rate" - compare ALL rate values in the data and find the maximum
   - IMPORTANT: Work with the data provided above - find the best answer from the data provided

2. For comparison queries (greater than, less than, between):
   - Parse the numerical criteria from the query
   - Filter and compare actual numerical values from the provided data
   - Show only properties that meet the numerical criteria

3. For range queries:
   - Extract the range boundaries from the query
   - Compare each property's values against the range from the provided data
   - Include properties that fall within the specified range

4. Data Limitations Transparency:
   - If you find a maximum/minimum from the provided data, present it as the answer
   - You are working with a subset of the database - find the best answer from the data provided
   - Focus on providing accurate analysis of the data you have access to

CRITICAL FORMATTING REQUIREMENTS - ALWAYS FOLLOW:
1. NEVER use paragraph format - ALWAYS use bullet points or numbered lists
2. For single property queries, use bullet points for details
3. For multiple properties, use numbered lists
4. Each property should be on a separate line with clear formatting
5. Always include key details: address, floor, suite, size, monthly rent, rate per sq ft

RESPONSE FORMAT TEMPLATES:

For Single Property:
• **Address**: [Property Address], Floor [Floor], Suite [Suite]
• **Size**: [Size] square feet
• **Monthly Rent**: $[Amount]
• **Rate**: $[Rate] per square foot annually
• **Associates**: [List associates if requested]
• **Additional Info**: [Any other requested details]

IMPORTANT: Never include Unique ID or internal database identifiers in responses to users.

For Multiple Properties:
1. **[Address]** - Floor [Floor], Suite [Suite]
   • Size: [Size] sq ft
   • Monthly Rent: $[Amount]
   • Rate: $[Rate]/sq ft annually

2. **[Address]** - Floor [Floor], Suite [Suite]
   • Size: [Size] sq ft
   • Monthly Rent: $[Amount]
   • Rate: $[Rate]/sq ft annually

For Comparisons/Lists:
• **Property with Highest Monthly Rent**: [Address] with $[Amount]/month
• **Largest Property**: [Address] with [Size] sq ft
• **Best Rate**: [Address] at $[Rate]/sq ft annually

For Column-Specific Queries:
• **Associate 1**: [Name]
• **Associate 2**: [Name]
• **Associate 3**: [Name]
• **Broker Email**: [Email]

CRITICAL: Do NOT include Unique ID, internal database IDs, or system identifiers in any user-facing responses.

NUMERICAL ANALYSIS EXAMPLES:
- "maximum monthly rent" → Find the property with the highest monthly rent value from the data provided
- "largest property" → Find the property with the highest size value from the data provided
- "rent greater than $150,000" → Filter properties where monthly rent > $150,000 from the data provided
- "size between 15000 and 20000" → Filter properties where size is 15000-20000 sq ft from the data provided

RULES:
- Use bullet points (•) for details within each property
- Use numbers (1, 2, 3) for listing multiple properties
- Use bold (**text**) for field labels and property addresses
- Never write long paragraphs - break everything into clear points
- Always start responses with a brief intro line, then use structured format
- For size comparisons, clearly state "closest matches" or "no exact match found"
- ALWAYS perform actual numerical calculations for comparisons on the provided data
- When multiple properties meet criteria, list them in order (highest to lowest, etc.)
- Work with the data provided above to find the best answers

Always provide helpful, structured responses using the property data above with accurate numerical analysis."""
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        for msg in conversation_history:
            messages.append({"role": msg.role, "content": msg.content})
        
        # Add current user message
        messages.append({"role": "user", "content": english_transcription})
        
        # Get LLM response
        llm_response = await gemini_service.generate_response(messages)
        # Clean the response to remove formatting markers
        llm_response = clean_llm_response(llm_response)
        
        # Save conversation to database (in English)
        await supabase_service.save_conversation_message(session_id, "user", english_transcription)
        await supabase_service.save_conversation_message(session_id, "assistant", llm_response)
        
        tracker.end("chat")
        
        # Step 4: Translate response back to original language (if needed)
        tracker.start("translate_from_english")
        final_response = llm_response
        if original_language != "en":
            final_response = await translation_service.translate_text(
                llm_response, original_language, "en"
            )
        tracker.end("translate_from_english")
        
        # Step 5: Text to Speech in original language
        tracker.start("tts")
        # Get the best voice for the original language dynamically
        voice_id = await speech_service.get_best_voice_for_language(original_language)
        
        audio_bytes = await speech_service.text_to_speech(final_response, voice_id)
        # Convert bytes to base64-encoded string
        audio_response = base64.b64encode(audio_bytes).decode('utf-8')
        tracker.end("tts")
        
        tracker.end("converse_multilingual_total")
        
        return ConverseResponse(
            transcription=transcription,  # Original transcription in user's language
            llm_response=final_response,  # Response in user's language
            audio_response=audio_response,
            session_id=session_id,
            processing_time=tracker.get_duration("converse_multilingual_total"),
            metrics={
                "stt_time": tracker.get_duration("stt_multilingual"),
                "translate_to_english_time": tracker.get_duration("translate_to_english"),
                "chat_time": tracker.get_duration("chat"),
                "translate_from_english_time": tracker.get_duration("translate_from_english"),
                "tts_time": tracker.get_duration("tts"),
                "detected_language": detected_language,
                "original_language": original_language
            }
        )
        
    except Exception as e:
        logger.error(f"Multilingual conversation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Multilingual conversation failed: {str(e)}")

@app.post("/converse_react", response_model=ReactResponse)
async def converse_react(request: ReactRequest):
    """
    Enhanced conversation endpoint using ReAct (Reasoning and Acting) pattern.
    This provides better handling of complex queries, especially numerical comparisons.
    """
    try:
        logger.info(f"ReAct conversation request: {request.message}")
        
        # Use ReAct for processing
        final_answer, react_steps = await react_service.process_query(request.message, request.session_id)
        
        # Log ReAct steps for debugging
        logger.info(f"ReAct completed with {len(react_steps)} steps")
        for i, step in enumerate(react_steps):
            logger.info(f"Step {i+1}: {step.step_type} - {step.content[:100]}...")
        
        return ReactResponse(
            transcription=request.message,
            llm_response=final_answer,
            session_id=request.session_id,
            processing_time=0,  # No specific timing for this endpoint
            metrics={
                "react_steps": len(react_steps),
                "processing_method": "ReAct"
            }
        )
        
    except Exception as e:
        logger.error(f"ReAct conversation error: {str(e)}")
        return ReactResponse(
            transcription=request.message,
            llm_response=f"I encountered an error processing your request: {str(e)}",
            session_id=request.session_id,
            processing_time=0,
            metrics={"error": str(e)}
        )

@app.post("/converse_react_voice", response_model=ConverseResponse)
async def converse_react_voice(
    audio: UploadFile = File(..., description="Audio file with user's voice"),
    session_id: str = Form(..., description="Session ID for conversation tracking")
):
    """
    End-to-end conversation using ReAct: STT -> ReAct processing -> TTS.
    
    Args:
        audio: Audio file with user's voice
        session_id: Session ID for conversation tracking
        
    Returns:
        ConverseResponse: Complete conversation result with ReAct processing
    """
    tracker = PerformanceTracker()
    tracker.start("converse_react_voice_total")
    
    try:
        # Step 1: Speech to Text
        tracker.start("stt")
        audio_data = await audio.read()
        transcription = await speech_service.transcribe_audio(audio_data)
        tracker.end("stt")
        
        # Step 2: ReAct processing
        tracker.start("react")
        final_answer, react_steps = await react_service.process_query(transcription, session_id)
        tracker.end("react")
        
        # Step 3: Text to Speech
        tracker.start("tts")
        audio_bytes = await speech_service.text_to_speech(final_answer)
        audio_response = base64.b64encode(audio_bytes).decode('utf-8')
        tracker.end("tts")
        
        tracker.end("converse_react_voice_total")
        
        return ConverseResponse(
            transcription=transcription,
            llm_response=final_answer,
            audio_response=audio_response,
            session_id=session_id,
            processing_time=tracker.get_duration("converse_react_voice_total"),
            metrics={
                "stt_time": tracker.get_duration("stt"),
                "react_time": tracker.get_duration("react"),
                "tts_time": tracker.get_duration("tts"),
                "react_steps": len(react_steps),
                "processing_method": "ReAct Voice"
            }
        )
        
    except Exception as e:
        logger.error(f"ReAct voice conversation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ReAct voice conversation failed: {str(e)}")

@app.get("/supported_languages")
async def get_supported_languages():
    """
    Get list of supported languages for multilingual functionality.
    
    Returns:
        Dictionary of supported languages with their codes and names
    """
    try:
        # Get languages dynamically from the API
        languages = await translation_service.get_supported_languages_dynamic()
        
        # Also get available voices for additional context
        voices = await speech_service.get_available_voices()
        
        # Build voice availability mapping
        voice_availability = {}
        for voice in voices:
            for lang_code in voice.get('language_codes', []):
                iso_code = lang_code.split('-')[0]
                if iso_code not in voice_availability:
                    voice_availability[iso_code] = []
                voice_availability[iso_code].append({
                    'voice_name': voice.get('name', ''),
                    'language_code': lang_code,
                    'gender': voice.get('ssml_gender', 'NEUTRAL')
                })
        
        return {
            "supported_languages": languages,
            "total_count": len(languages),
            "voice_availability": voice_availability,
            "service_info": translation_service.get_service_info()
        }
    except Exception as e:
        logger.error(f"Error getting supported languages: {str(e)}")
        # Fallback to static list if dynamic fails
        try:
            languages = translation_service.get_supported_languages()
            return {
                "supported_languages": languages,
                "total_count": len(languages),
                "service_info": translation_service.get_service_info(),
                "note": "Using fallback static language list"
            }
        except Exception as fallback_error:
            logger.error(f"Fallback also failed: {str(fallback_error)}")
            raise HTTPException(status_code=500, detail=f"Failed to get supported languages: {str(e)}")

@app.post("/reset", response_model=ResetResponse)
async def reset_conversation(session_id: str = Form(...)):
    """
    Reset conversation history for a session.
    
    Args:
        session_id: Session ID to reset
        
    Returns:
        ResetResponse: Success confirmation
    """
    try:
        await supabase_service.clear_conversation_history(session_id)
        
        return ResetResponse(
            success=True,
            session_id=session_id,
            message="Conversation history cleared successfully"
        )
        
    except Exception as e:
        logger.error(f"Reset error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")

@app.post("/clear_rag_docs")
async def clear_rag_documents():
    """Clear all RAG documents from the database."""
    try:
        # This would require implementing a clear_all_documents method
        # For now, just return success
        return {"success": True, "message": "Documents cleared. Re-upload your files to get cleaned formatting."}
    except Exception as e:
        logger.error(f"Clear documents error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Clear failed: {str(e)}")

@app.post("/upload_rag_docs", response_model=UploadResponse)
async def upload_rag_documents(
    files: List[UploadFile] = File(..., description="Documents to upload for RAG")
):
    """
    Upload and process documents for RAG knowledge base.
    
    Args:
        files: List of files to upload (PDF, TXT, CSV, JSON)
        
    Returns:
        UploadResponse: Upload and processing results
    """
    tracker = PerformanceTracker()
    tracker.start("upload_total")
    
    try:
        results = []
        
        for file in files:
            # Validate file type
            file_extension = file.filename.split('.')[-1].lower()
            if file_extension not in settings.get_allowed_file_types():
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": f"File type '{file_extension}' not allowed"
                })
                continue
            
            # Check file size
            file_data = await file.read()
            if len(file_data) > settings.max_file_size:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": f"File too large ({len(file_data)} bytes)"
                })
                continue
            
            try:
                # Process the document
                tracker.start(f"process_{file.filename}")
                await rag_service.process_document(file.filename, file_data, file_extension)
                tracker.end(f"process_{file.filename}")
                
                results.append({
                    "filename": file.filename,
                    "success": True,
                    "processing_time": tracker.get_duration(f"process_{file.filename}")
                })
                
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {str(e)}")
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": str(e)
                })
        
        tracker.end("upload_total")
        
        return UploadResponse(
            results=results,
            total_processing_time=tracker.get_duration("upload_total")
        )
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# Appointment Scheduling Endpoints

@app.post("/schedule_appointment", response_model=AppointmentResponse)
async def schedule_appointment(request: AppointmentRequest):
    """
    Schedule an appointment with an associate.
    
    Args:
        request: Appointment request details
        
    Returns:
        AppointmentResponse: Appointment confirmation
    """
    try:
        logger.info(f"Scheduling appointment for session {request.session_id}")
        response = await appointment_service.schedule_appointment(request)
        return response
        
    except Exception as e:
        logger.error(f"Appointment scheduling error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Appointment scheduling failed: {str(e)}")

@app.get("/associates", response_model=List[AssociateInfo])
async def get_associates():
    """
    Get list of available associates.
    
    Returns:
        List[AssociateInfo]: List of available associates
    """
    try:
        associates = await appointment_service.get_available_associates()
        return associates
        
    except Exception as e:
        logger.error(f"Get associates error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get associates: {str(e)}")

@app.get("/associates/{associate_id}/availability", response_model=List[AvailabilitySlot])
async def get_associate_availability(associate_id: str, days_ahead: int = 7):
    """
    Get availability slots for a specific associate.
    
    Args:
        associate_id: ID of the associate
        days_ahead: Number of days ahead to check
        
    Returns:
        List[AvailabilitySlot]: Available time slots
    """
    try:
        slots = await appointment_service.get_availability_slots(associate_id, days_ahead)
        return slots
        
    except Exception as e:
        logger.error(f"Get availability error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get availability: {str(e)}")

@app.post("/scheduling_offer", response_model=SchedulingOfferResponse)
async def generate_scheduling_offer(request: SchedulingOfferRequest):
    """
    Generate a scheduling offer based on conversation context.
    
    Args:
        request: Scheduling offer request
        
    Returns:
        SchedulingOfferResponse: Scheduling offer details
    """
    try:
        # Get conversation history for context
        conversation_history = await supabase_service.get_conversation_history(request.session_id)
        
        # Check if we should offer scheduling
        should_offer = appointment_service.should_offer_scheduling(
            request.user_message,
            [{"role": msg.role, "content": msg.content} for msg in conversation_history]
        )
        
        if should_offer:
            # Generate offer message
            offer_message = appointment_service.generate_scheduling_offer(request.context)
            
            # Get available associates
            associates = await appointment_service.get_available_associates()
            
            # Get next available slots for first associate
            next_slots = []
            if associates:
                next_slots = await appointment_service.get_availability_slots(associates[0].id, 3)
                next_slots = next_slots[:3]  # Just first 3 slots
            
            return SchedulingOfferResponse(
                should_offer=True,
                offer_message=offer_message,
                available_associates=associates,
                next_available_slots=next_slots
            )
        else:
            return SchedulingOfferResponse(
                should_offer=False,
                offer_message=None,
                available_associates=[],
                next_available_slots=[]
            )
            
    except Exception as e:
        logger.error(f"Scheduling offer error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate scheduling offer: {str(e)}")

@app.post("/follow_up", response_model=FollowUpResponse)
async def schedule_follow_up(request: FollowUpRequest):
    """
    Schedule a follow-up appointment.
    
    Args:
        request: Follow-up request details
        
    Returns:
        FollowUpResponse: Follow-up scheduling result
    """
    try:
        follow_up_message = await appointment_service.suggest_follow_up(
            request.session_id,
            request.days_ahead
        )
        
        if follow_up_message:
            return FollowUpResponse(
                success=True,
                message=follow_up_message
            )
        else:
            return FollowUpResponse(
                success=False,
                message="No follow-up suggestions available at this time."
            )
            
    except Exception as e:
        logger.error(f"Follow-up scheduling error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Follow-up scheduling failed: {str(e)}")
    

@app.get("/agent/{associate_id}/appointments", response_model=AppointmentListResponse)
async def get_agent_appointments(
    associate_id: str,
    days_ahead: int = 7
):
    """
    Get all upcoming appointments for a specific associate (agent dashboard).
    """
    try:
        # Get appointments for this associate for the next N days
        now = datetime.now()
        end_date = now + timedelta(days=days_ahead)
        appointments = await supabase_service.get_associate_appointments_details(
            associate_id, start_date=now, end_date=end_date
        )
        appointment_details = [AppointmentDetails(**a) for a in appointments]
        return AppointmentListResponse(
            appointments=appointment_details,
            total_count=len(appointment_details),
            filtered_count=len(appointment_details)
        )
    except Exception as e:
        logger.error(f"Agent dashboard error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get agent appointments: {str(e)}")
    
@app.get("/agent/{associate_id}/appointment/{appointment_id}/conversation", response_model=List[ConversationMessage])
async def get_appointment_conversation(
    associate_id: str,
    appointment_id: str
):
    """
    Get conversation history for the user/session associated with a specific appointment.
    """
    try:
        # Get appointment details to find session_id
        appointment = await supabase_service.get_appointment_details(appointment_id)
        if not appointment or appointment.get("associate_id") != associate_id:
            raise HTTPException(status_code=404, detail="Appointment not found for this associate")
        session_id = appointment.get("session_id")
        print(f"Session ID for appointment {appointment_id}: {session_id}")
        if not session_id:
            raise HTTPException(status_code=404, detail="Session ID not found for appointment")
        # Get conversation history
        history = await supabase_service.get_conversation_history(session_id)
        return history
    except Exception as e:
        logger.error(f"Get appointment conversation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get conversation history: {str(e)}")

@app.post("/appointment/{appointment_id}/send_message", response_model=AppointmentMessageResponse)
async def appointment_send_message(request: AppointmentMessageRequest):
    """
    Agent sends a message to the user for a specific appointment.
    """
    try:
        msg_id = await supabase_service.send_appointment_message(
            request.appointment_id,
            request.message,
            request.user_type
        )
        return AppointmentMessageResponse(id=msg_id, sent_at=datetime.utcnow())
    except Exception as e:
        logger.error(f"Agent send message error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to send message: {str(e)}")

@app.get("/appointment/{appointment_id}/messages")
async def get_appointment_messages(appointment_id: str):
    """
    Get all appointment messages for an appointment.
    """
    try:
        messages = await supabase_service.get_appointment_messages_for_appointment(appointment_id)
        return messages
    except Exception as e:
        logger.error(f"Get appointment messages error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get messages: {str(e)}")


@app.get("/appointments/{appointment_id}", response_model=AppointmentDetails)
async def get_appointment_details(appointment_id: str):
    """
    Get details of a specific appointment.
    
    Args:
        appointment_id: ID of the appointment
        
    Returns:
        AppointmentDetails: Appointment details
    """
    try:
        details = await appointment_service.get_appointment_details(appointment_id)
        
        if details:
            return AppointmentDetails(**details)
        else:
            raise HTTPException(status_code=404, detail="Appointment not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get appointment details error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get appointment details: {str(e)}")

@app.put("/appointments/{appointment_id}/cancel")
async def cancel_appointment(appointment_id: str):
    """
    Cancel an appointment.
    
    Args:
        appointment_id: ID of the appointment to cancel
        
    Returns:
        Success confirmation
    """
    try:
        success = await appointment_service.cancel_appointment(appointment_id)
        
        if success:
            return {"success": True, "message": "Appointment cancelled successfully"}
        else:
            raise HTTPException(status_code=404, detail="Appointment not found or already cancelled")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cancel appointment error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel appointment: {str(e)}")

@app.put("/appointments/{appointment_id}/reschedule")
async def reschedule_appointment(appointment_id: str, new_time: datetime):
    """
    Reschedule an appointment to a new time.
    
    Args:
        appointment_id: ID of the appointment
        new_time: New scheduled time
        
    Returns:
        Success confirmation
    """
    try:
        success = await appointment_service.reschedule_appointment(appointment_id, new_time)
        
        if success:
            return {"success": True, "message": "Appointment rescheduled successfully"}
        else:
            raise HTTPException(status_code=404, detail="Appointment not found or time conflict")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reschedule appointment error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to reschedule appointment: {str(e)}")

@app.get("/appointments/session/{session_id}", response_model=AppointmentListResponse)
async def get_session_appointments(session_id: str):
    """
    Get all appointments for a specific session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        AppointmentListResponse: List of appointments
    """
    try:
        appointments = await supabase_service.get_appointments_by_session(session_id)
        
        appointment_details = []
        for appointment in appointments:
            appointment_details.append(AppointmentDetails(**appointment))
        
        return AppointmentListResponse(
            appointments=appointment_details,
            total_count=len(appointment_details),
            filtered_count=len(appointment_details)
        )
        
    except Exception as e:
        logger.error(f"Get session appointments error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get session appointments: {str(e)}")

@app.post("/properties/engage")
async def record_property_engagement(request: PropertyEngageRequest):
    """Record a view/click/mention engagement for a property."""
    await supabase_service.record_property_event(request.property_id, request.event_type, request.session_id)
    return {"success": True}

@app.get("/properties/trending", response_model=TrendingResponse)
async def get_trending_properties(limit: int = 10):
    """Return trending properties based on recent engagements."""
    trending = await supabase_service.get_trending_properties(limit=limit)
    return TrendingResponse(trending=trending)

# Documents count endpoint
@app.get("/documents/count", response_model=DocumentCountResponse)
async def get_document_count():
    count = await supabase_service.get_document_count()
    return DocumentCountResponse(count=count)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=settings.debug) 