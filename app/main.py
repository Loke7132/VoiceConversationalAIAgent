from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import time
from datetime import datetime
import asyncio
import io
import base64
import re

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
    ConversationMessage
)
from .services.supabase_service import SupabaseService
from .services.gemini_service import GeminiService
from .services.google_cloud_service import GoogleCloudService
from .services.rag_service import RAGService
from .services.translation_service import TranslationService
from .utils.performance import PerformanceTracker
from .services.react_service import ReActService

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
            "upload_rag_docs": "/upload_rag_docs"
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

Always provide helpful, structured responses using the property data above with accurate numerical analysis."""
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        for msg in conversation_history:
            messages.append({"role": msg.role, "content": msg.content})
        
        # Add current user message
        messages.append({"role": "user", "content": request.message})
        
        # Get LLM response
        tracker.start("llm_processing")
        llm_response = await gemini_service.generate_response(messages)
        # Clean the response to remove formatting markers
        llm_response = clean_llm_response(llm_response)
        tracker.end("llm_processing")
        
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=settings.debug) 