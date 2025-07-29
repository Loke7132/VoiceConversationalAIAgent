from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta, timezone
import logging
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
import re
import difflib
from pathlib import Path

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from ..config import settings
from ..models import AppointmentRequest, AppointmentResponse, AvailabilitySlot, AssociateInfo

logger = logging.getLogger(__name__)


class AppointmentService:
    """Service for handling intelligent appointment scheduling."""
    
    def __init__(self, supabase_service):
        self.supabase_service = supabase_service
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.calendar_service = None
        self.associate_keywords = [
            'associate', 'agent', 'broker', 'representative', 'advisor',
            'consultant', 'specialist', 'expert', 'team member', 'staff',
            'contact', 'speak with', 'talk to', 'meet with', 'call with',
            'appointment', 'schedule', 'book', 'arrange', 'meeting'
        ]
        self.scheduling_keywords = [
            'schedule', 'book', 'appointment', 'meeting', 'call', 'available',
            'when can', 'free time', 'availability', 'calendar', 'time slot',
            'arrange', 'set up', 'plan', 'organize', 'follow up', 'follow-up'
        ]
        
        # Track conversation state for appointment booking
        self.pending_appointments = {}  # session_id -> appointment_state
        # Track pending associate name clarifications (session_id ➜ details)
        self.pending_associate_confirmations: Dict[str, Dict[str, Any]] = {}
        # Track contact details already provided by the user (session_id -> dict)
        self.pending_contact_details: Dict[str, Dict[str, str]] = {}
        
        logger.info("Appointment service initialized")
    
    async def _run_sync(self, func, *args, **kwargs):
        """Run synchronous function in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args, **kwargs)
    
    def parse_contact_details(self, message: str) -> Optional[Dict[str, str]]:
        """
        Parse contact details from a user message.
        
        Args:
            message: User message that might contain contact details
            
        Returns:
            Dict with name, phone, email if found, or None
        """
        # Common patterns for contact details
        phone_pattern = r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'
        # More flexible email pattern to handle partial domains
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+(?:\.[A-Z|a-z]{2,}|[0-9]+)\b'
        
        # Find phone and email
        phone_match = re.search(phone_pattern, message)
        email_match = re.search(email_pattern, message)
        
        if phone_match or email_match:
            phone = phone_match.group(0) if phone_match else None
            email = email_match.group(0) if email_match else None
            
            # Extract name (usually the first part before phone/email)
            name = None
            
            # Try to extract name from common patterns
            # Pattern 1: "Name, phone, email"
            parts = [part.strip() for part in message.split(',')]
            if len(parts) >= 2:
                # First part is likely the name if it doesn't contain phone/email
                potential_name = parts[0]
                if not re.search(phone_pattern, potential_name) and not re.search(email_pattern, potential_name):
                    name = potential_name
            else:
                # Pattern 2: "Name phone email" (space separated)
                words = message.split()
                if len(words) >= 2:
                    # Take first word(s) that don't match phone/email patterns
                    name_parts = []
                    for word in words:
                        if not re.search(phone_pattern, word) and not re.search(email_pattern, word):
                            name_parts.append(word)
                        else:
                            break
                    if name_parts:
                        name = ' '.join(name_parts)
            
            details = {
                'name': name or 'User',
                'phone': phone,
                'email': email
            }
            # Cache details so we don't ask for them again in the same session
            # (they may be needed later if the flow continues)
            return details
        
        return None
    
    def detect_direct_booking_request(self, message: str) -> bool:
        """
        Detect if user is making a direct booking request.
        
        Args:
            message: Current user message
            
        Returns:
            bool: True if this is a direct booking request
        """
        message_lower = message.lower()
        
        # Direct booking keywords
        direct_booking_keywords = [
            'book an appointment',
            'book appointment',
            'schedule an appointment', 
            'schedule appointment',
            'book a meeting',
            'schedule a meeting',
            'book with',
            'schedule with',
            'make an appointment',
            'set up appointment',
            'arrange appointment'
        ]
        
        return any(keyword in message_lower for keyword in direct_booking_keywords)
    
    def detect_appointment_confirmation(self, message: str, conversation_history: List[Dict]) -> bool:
        """
        Detect if user is confirming an appointment time.
        
        Args:
            message: Current user message
            conversation_history: Recent conversation history
            
        Returns:
            bool: True if this appears to be appointment confirmation
        """
        message_lower = message.lower()
        
        # Basic affirmative responses
        affirmative_keywords = [
            'yes', 'yeah', 'sure', 'ok', 'okay', 'sounds good', 'that works',
            'perfect', 'great'
        ]

        has_confirmation = any(re.search(rf'\b{re.escape(kw)}\b', message_lower) for kw in affirmative_keywords)

        # Additionally, if the user explicitly provides a day/time preference, treat that as confirmation
        if not has_confirmation:
            if self.extract_time_preference(message):
                has_confirmation = True
        
        # Check if recent assistant message offered scheduling
        recent_assistant_messages = [
            msg for msg in conversation_history[-3:] 
            if msg.get('role') == 'assistant'
        ]
        
        offered_scheduling = any(
            'schedule' in msg.get('content', '').lower() or 
            'appointment' in msg.get('content', '').lower() or
            'meeting' in msg.get('content', '').lower() or
            'availability' in msg.get('content', '').lower()
            for msg in recent_assistant_messages
        )
        
        return has_confirmation and offered_scheduling

    # ---------------- Associate name clarification helpers -----------------

    def _best_associate_match(self, input_name: str, associates: List[AssociateInfo]) -> Tuple[Optional[AssociateInfo], float]:
        """Return best fuzzy match for a name and its similarity score (0-1)."""
        best: Optional[AssociateInfo] = None
        best_score = 0.0
        for a in associates:
            score = difflib.SequenceMatcher(None, input_name.lower(), a.name.lower()).ratio()
            if score > best_score:
                best_score = score
                best = a
        return best, best_score

    def _maybe_clarify_associate(self, input_name: str, session_id: str) -> Optional[str]:
        """
        If the input name is only a moderate fuzzy match (0.60-0.85) ask the user
        for confirmation and store pending state. Return clarification prompt or None.
        """
        try:
            associates = asyncio.run(self.get_available_associates())  # safe – short list
            best, score = self._best_associate_match(input_name, associates)
            if best is None:
                return None

            if score >= 0.85:
                # High confidence – treat as confirmed. Inject into pending mapping so later code can use it.
                self.pending_associate_confirmations.pop(session_id, None)
                if session_id in self.pending_appointments:
                    self.pending_appointments[session_id]['associate_id'] = best.id
                return None

            if score >= 0.6:
                # Ask user to confirm
                self.pending_associate_confirmations[session_id] = {
                    'suggested_name': best.name,
                    'suggested_id': best.id,
                }
                return f"Did you mean {best.name}? (yes / no)"
            return None
        except Exception as e:
            logger.warning(f"Clarification helper error: {str(e)}")
            return None

    def _is_affirmative(self, message: str) -> bool:
        """Rough check whether the message is an affirmative yes."""
        return re.search(r"\b(yes|yeah|yep|correct|right|sure|y|ok|okay)\b", message.lower()) is not None

    # ---------------- Google Calendar OAuth helpers -----------------

    def _get_oauth_creds(self):
        """Load cached token or run OAuth flow to obtain user credentials."""
        SCOPES = ["https://www.googleapis.com/auth/calendar"]
        token_path = Path(settings.token_json_path)

        creds: Optional[Credentials] = None
        if token_path.exists():
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)

        # Refresh if expired
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())

        # First-time authorization
        if not creds or not creds.valid:
            flow = InstalledAppFlow.from_client_config(
                {
                    "installed": {
                        "client_id": settings.google_oauth_client_id,
                        "client_secret": settings.google_oauth_client_secret,
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                        "redirect_uris": ["http://localhost"]
                    }
                },
                SCOPES,
            )
            # Run the local server flow which opens a browser for the user and listens
            # on an ephemeral localhost port for the redirect. This method replaces
            # the deprecated out-of-band (OOB) flow that Google no longer supports.
            creds = flow.run_local_server(
                port=0,
                authorization_prompt_message="\n===== Google OAuth consent required =====\nYour browser will open with a Google consent screen.\nPlease log in with the calendar owner account and approve access.",
                success_message="Authentication complete. You may now close this tab and return to the app.",
            )
            # Cache for next runs
            token_path.write_text(creds.to_json())

        return creds
    
    def extract_time_preference(self, message: str) -> Optional[str]:
        """
        Extract time preference from user message.
        
        Args:
            message: User message
            
        Returns:
            str: Time preference if found
        """
        message_lower = message.lower()
        
        # Common time patterns
        time_patterns = [
            r'tomorrow at (\d{1,2}(?::\d{2})?\s*(?:am|pm))',
            r'today at (\d{1,2}(?::\d{2})?\s*(?:am|pm))',
            r'(\d{1,2}(?::\d{2})?\s*(?:am|pm))',
            r'tomorrow',
            r'today'
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, message_lower)
            if match:
                return match.group(1) if match.groups() else match.group(0)
        
        return None
    
    async def handle_appointment_flow(self, message: str, session_id: str, conversation_history: List[Dict]) -> Optional[str]:
        """
        Handle the appointment booking flow automatically.
        
        Args:
            message: User message
            session_id: Session identifier
            conversation_history: Recent conversation history
            
        Returns:
            str: Response message if appointment was handled, None otherwise
        """
        try:
            # Check if this is a direct booking request
            if self.detect_direct_booking_request(message):
                # Extract associate and time from the message
                associate_name = self.parse_associate_from_message(message)
                scheduled_time = self.parse_time_from_message(message)
                
                # Attempt clarification if associate name provided
                if associate_name:
                    clarification = self._maybe_clarify_associate(associate_name, session_id)
                    if clarification:
                        # Pause flow until user confirms
                        return clarification

                # Store appointment state for this session
                self.pending_appointments[session_id] = {
                    'direct_booking': True,
                    'associate_name': associate_name,
                    'time_preference': None,
                    'preferred_time': scheduled_time
                }

                # Ask for contact details immediately
                response = "I'd be happy to help you book that appointment! Please share your name, phone number, and email address so I can schedule it for you."

                if associate_name:
                    response += f" I'll make sure to book it with {associate_name}."

                if scheduled_time:
                    response += f" I'll schedule it for {scheduled_time.strftime('%A, %B %d at %I:%M %p')}."

                return response
            
            # Check if user is providing contact details
            contact_details = self.parse_contact_details(message)
            
            if contact_details:
                logger.info(f"Contact details parsed: {contact_details}")
                
                # Always remember the latest contact details provided by the user for this session
                self.pending_contact_details[session_id] = contact_details
                
                # Check if there's a pending appointment for this session
                if session_id in self.pending_appointments:
                    appointment_state = self.pending_appointments[session_id]
                    
                    # Create the appointment
                    try:
                        # Get scheduled time first
                        scheduled_time = appointment_state.get('preferred_time')
                        if not scheduled_time:
                            scheduled_time = self.parse_time_from_message(message)
                        
                        if not scheduled_time:
                            # Default to tomorrow at 10 AM
                            scheduled_time = datetime.now(timezone.utc) + timedelta(days=1)
                            scheduled_time = scheduled_time.replace(hour=10, minute=0, second=0, microsecond=0)
                        
                        # Build list of preferred associates from conversation context
                        preferred_associates = []
                        
                        # Add associate from state
                        if appointment_state.get('associate_name'):
                            preferred_associates.append(appointment_state.get('associate_name'))
                        
                        # Add associate from current message
                        associate_name_from_message = self.parse_associate_from_message(message)
                        if associate_name_from_message:
                            preferred_associates.append(associate_name_from_message)
                        
                        # Add associates from conversation context
                        associate_name_from_context = self.extract_associate_from_context(conversation_history)
                        if associate_name_from_context:
                            preferred_associates.append(associate_name_from_context)
                        
                        # Remove duplicates while preserving order
                        seen = set()
                        unique_preferred = []
                        for name in preferred_associates:
                            if name and name not in seen:
                                seen.add(name)
                                unique_preferred.append(name)
                        
                        logger.info(f"Preferred associates: {unique_preferred}")
                        
                        # Find available associate
                        associate_id = await self.find_available_associate_for_time(unique_preferred, scheduled_time)
                        
                        if not associate_id:
                            return f"I apologize, but none of the requested associates are available at {scheduled_time.strftime('%A, %B %d at %I:%M %p')}. Would you like me to suggest an alternative time or associate?"
                        
                        # Create appointment request
                        appointment_request = AppointmentRequest(
                            session_id=session_id,
                            associate_id=associate_id,
                            user_name=contact_details['name'],
                            user_email=contact_details['email'],
                            user_phone=contact_details['phone'],
                            scheduled_time=scheduled_time,
                            appointment_type="consultation",
                            notes=f"Auto-scheduled from conversation. Time preference: {appointment_state.get('time_preference', 'flexible')}. Preferred associates: {unique_preferred}. Used: {associate_id}"
                        )
                        
                        # Schedule the appointment
                        response = await self.schedule_appointment(appointment_request)
                        
                        # Clear pending appointment
                        del self.pending_appointments[session_id]
                        
                        if response.success:
                            confirmation = f"Perfect! I've scheduled your appointment for {scheduled_time.strftime('%A, %B %d at %I:%M %p')} with {response.associate_name}. "
                            # If the booked associate isn’t in the original preferred list, add an explanatory note
                            if response.associate_name:
                                booked_lower = response.associate_name.lower()
                                # Determine if the booked associate matches any preferred name (case-insensitive, substring tolerant)
                                matches_preferred = any(
                                    self._names_match(pref, response.associate_name)
                                    for pref in unique_preferred
                                )
                                if not matches_preferred:
                                    confirmation += "Your requested associate was unavailable at that time, so I booked with an available colleague. "

                            confirmation += f"You'll receive a calendar invitation at {contact_details['email']} to confirm the details."
                            # Clear cached contact details as well
                            self.pending_contact_details.pop(session_id, None)
                            return confirmation
                        else:
                            return f"I encountered an issue scheduling your appointment: {response.message}. Please try again or contact us directly."
                            
                    except Exception as e:
                        logger.error(f"Error creating appointment: {str(e)}")
                        return "I apologize, but I encountered an error while scheduling your appointment. Please try again or contact us directly."
                
                else:
                    # No pending appointment, but user provided contact details
                    # This might be a direct booking attempt
                    return "Thank you for providing your contact details. Let me check our availability and schedule an appointment for you."
            
            # Handle pending associate clarification
            if session_id in self.pending_associate_confirmations:
                if self._is_affirmative(message):
                    confirm = self.pending_associate_confirmations.pop(session_id)
                    # Inject confirmed associate into pending appointment if exists
                    if session_id in self.pending_appointments:
                        self.pending_appointments[session_id]['associate_name'] = confirm['suggested_name']
                        self.pending_appointments[session_id]['associate_id'] = confirm['suggested_id']
                        # Proceed as though user just provided affirmative; ask for contact details
                        return "Great! Please share your name, phone number, and email address so I can finalize the booking."
                # If negative or unclear, ask again
                if re.search(r"\b(no|nah|negative|not)\b", message.lower()):
                    # Clear pending and ask user to specify full name
                    self.pending_associate_confirmations.pop(session_id, None)
                    return "No problem — could you please provide the full name of the associate you’d like to meet with?"

            # Check if user is confirming appointment time
            elif self.detect_appointment_confirmation(message, conversation_history):
                # Extract user preferences from the confirmation message
                time_preference = self.extract_time_preference(message)
                preferred_time = self.parse_time_from_message(message)
                
                # If we only extracted a time (e.g., "1 pm") without a date, attempt to parse it assuming tomorrow
                if not preferred_time and time_preference:
                    preferred_time = self._parse_time_preference(time_preference)

                associate_name = self.parse_associate_from_message(message)

                # Check if we already have contact details from earlier in the conversation
                existing_details = self.pending_contact_details.get(session_id)

                self.pending_appointments[session_id] = {
                    'confirmed': True,
                    'time_preference': time_preference,
                    'preferred_time': preferred_time,
                    'associate_name': associate_name,
                    'contact_details': existing_details if existing_details else None
                }

                if existing_details:
                    try:
                        # Determine scheduled_time
                        scheduled_time = preferred_time
                        if not scheduled_time:
                            scheduled_time = datetime.now(timezone.utc) + timedelta(days=1)
                            scheduled_time = scheduled_time.replace(hour=10, minute=0, second=0, microsecond=0)

                        # Build list of preferred associates from conversation context
                        preferred_associates = []
                        if associate_name:
                            preferred_associates.append(associate_name)
                        associate_name_from_context = self.extract_associate_from_context(conversation_history)
                        if associate_name_from_context:
                            preferred_associates.append(associate_name_from_context)

                        # Remove duplicates
                        seen = set(); unique_preferred = []
                        for n in preferred_associates:
                            if n and n not in seen:
                                seen.add(n); unique_preferred.append(n)

                        # Find available associate
                        associate_id = await self.find_available_associate_for_time(unique_preferred, scheduled_time)
                        if not associate_id:
                            return f"I apologize, but none of the requested associates are available at {scheduled_time.strftime('%A, %B %d at %I:%M %p')}. Would you like me to suggest an alternative time or associate?"

                        # Create appointment request
                        appointment_request = AppointmentRequest(
                            session_id=session_id,
                            associate_id=associate_id,
                            user_name=existing_details['name'],
                            user_email=existing_details['email'],
                            user_phone=existing_details['phone'],
                            scheduled_time=scheduled_time,
                            appointment_type="consultation",
                            notes=f"Auto-scheduled from confirmation flow. Preferred associates: {unique_preferred}. Used: {associate_id}"
                        )

                        response = await self.schedule_appointment(appointment_request)

                        # Clear any pending state
                        self.pending_appointments.pop(session_id, None)
                        self.pending_contact_details.pop(session_id, None)

                        if response.success:
                            confirmation = f"Perfect! I've scheduled your appointment for {scheduled_time.strftime('%A, %B %d at %I:%M %p')} with {response.associate_name}. "
                            booked_lower = response.associate_name.lower()
                            matches_preferred = any(
                                self._names_match(pref, response.associate_name)
                                for pref in unique_preferred
                            )
                            if not matches_preferred:
                                confirmation += "Your requested associate was unavailable at that time, so I booked with an available colleague. "

                            confirmation += f"You'll receive a calendar invitation at {existing_details['email']} to confirm the details."
                            return confirmation
                        else:
                            return f"I encountered an issue scheduling your appointment: {response.message}. Please try again or contact us directly."
                    except Exception as e:
                        logger.error(f"Error scheduling from confirmation flow: {str(e)}")
                        return "I apologize, but I encountered an error while scheduling your appointment. Please try again or contact us directly." 
                else:
                    return "Excellent! Please share your name, phone number, and email address so I can schedule your appointment and send you a calendar invitation."
            
            return None
            
        except Exception as e:
            logger.error(f"Error in appointment flow: {str(e)}")
            return None
    
    def _parse_time_preference(self, time_preference: str) -> Optional[datetime]:
        """
        Parse time preference string into datetime object.
        
        Args:
            time_preference: Time preference string
            
        Returns:
            datetime object or None
        """
        try:
            now = datetime.now(timezone.utc)
            
            if 'tomorrow' in time_preference.lower():
                base_date = now + timedelta(days=1)
                
                # Extract time if present
                time_match = re.search(r'(\d{1,2})\s*(?::\s*(\d{2}))?\s*(am|pm)', time_preference.lower())
                if time_match:
                    hour = int(time_match.group(1))
                    minute = int(time_match.group(2)) if time_match.group(2) else 0
                    am_pm = time_match.group(3)
                    
                    if am_pm == 'pm' and hour != 12:
                        hour += 12
                    elif am_pm == 'am' and hour == 12:
                        hour = 0
                    
                    return base_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
                else:
                    return base_date.replace(hour=10, minute=0, second=0, microsecond=0)
            
            elif 'today' in time_preference.lower():
                base_date = now
                
                # Extract time if present
                time_match = re.search(r'(\d{1,2})\s*(?::\s*(\d{2}))?\s*(am|pm)', time_preference.lower())
                if time_match:
                    hour = int(time_match.group(1))
                    minute = int(time_match.group(2)) if time_match.group(2) else 0
                    am_pm = time_match.group(3)
                    
                    if am_pm == 'pm' and hour != 12:
                        hour += 12
                    elif am_pm == 'am' and hour == 12:
                        hour = 0
                    
                    return base_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
                else:
                    return base_date.replace(hour=15, minute=0, second=0, microsecond=0)
            
            # Handle time strings without explicit date (assume tomorrow)
            time_match = re.search(r'(\d{1,2})\s*(?::\s*(\d{2}))?\s*(am|pm)', time_preference.lower())
            if time_match:
                base_date = now + timedelta(days=1)
                hour = int(time_match.group(1))
                minute = int(time_match.group(2)) if time_match.group(2) else 0
                am_pm = time_match.group(3)

                if am_pm == 'pm' and hour != 12:
                    hour += 12
                elif am_pm == 'am' and hour == 12:
                    hour = 0

                return base_date.replace(hour=hour, minute=minute, second=0, microsecond=0)

            return None
            
        except Exception:
            return None
    
    def detect_associate_query(self, message: str) -> bool:
        """
        Detect if the user is asking about associates or expressing interest in scheduling.
        
        Args:
            message: User's message
            
        Returns:
            bool: True if associate-related query detected
        """
        message_lower = message.lower()
        
        # Check for associate keywords
        has_associate_keyword = any(keyword in message_lower for keyword in self.associate_keywords)
        
        # Check for scheduling intent keywords
        has_scheduling_keyword = any(keyword in message_lower for keyword in self.scheduling_keywords)
        
        # Check for question patterns that might indicate interest in talking to someone
        question_patterns = [
            r'who can help',
            r'who should i talk to',
            r'who can i speak',
            r'who handles',
            r'who is responsible',
            r'can someone help',
            r'need to talk',
            r'want to speak',
            r'like to meet',
            r'would like to discuss',
            r'interested in learning more',
            r'tell me more',
            r'more information',
            r'follow up',
            r'next steps'
        ]
        
        has_question_pattern = any(re.search(pattern, message_lower) for pattern in question_patterns)
        
        return has_associate_keyword or has_scheduling_keyword or has_question_pattern
    
    def should_offer_scheduling(self, message: str, conversation_history: List[Dict]) -> bool:
        """
        Determine if we should offer scheduling based on message and conversation context.
        
        Args:
            message: Current user message
            conversation_history: Recent conversation history
            
        Returns:
            bool: True if scheduling should be offered
        """
        try:
            logger.info(f"Checking scheduling offer for message: '{message}'")
            
            # Check if the user is NOW explicitly asking to schedule; if so, prioritise that request regardless of prior disclaimers.
            scheduling_keywords = ['schedule', 'book', 'appointment', 'meeting']
            has_scheduling_keywords = any(keyword in message.lower() for keyword in scheduling_keywords)

            if has_scheduling_keywords:
                logger.info("Direct scheduling request detected — will offer scheduling regardless of earlier messages")
                return True

            # Otherwise, see if we recently told them to contact the broker directly; in that case, don’t repeat the offer
            recent_assistant_messages = [
                msg for msg in conversation_history[-2:]
                if msg.get('role') == 'assistant'
            ]

            for msg in recent_assistant_messages:
                content = msg.get('content', '').lower()
                if any(phrase in content for phrase in [
                    'contact the broker',
                    'email at',
                    'reach out to',
                    'contact information',
                    'you can reach out to',
                    'via email',
                    'broker should be able to facilitate',
                    'do not have their direct contact',
                    'try reaching out to'
                ]):
                    logger.info("Assistant already provided specific contact instructions, not offering scheduling")
                    return False
            
            # Check if user is asking about associates
            associate_detected = self.detect_associate_query(message)
            logger.info(f"Associate query detected: {associate_detected}")
            
            if associate_detected:
                return True
            
            # Check conversation context - if recent messages mentioned properties or assistance
            recent_messages = conversation_history[-3:] if len(conversation_history) >= 3 else conversation_history
            
            for msg in recent_messages:
                if msg.get('role') == 'assistant':
                    content = msg.get('content', '').lower()
                    # If assistant recently provided property info, offer scheduling
                    if any(keyword in content for keyword in ['property', 'address', 'rent', 'size', 'available']):
                        logger.info("Property context detected in conversation history")
                        return True
            
            logger.info("No scheduling triggers detected")
            return False
        except Exception as e:
            logger.error(f"Error in should_offer_scheduling: {str(e)}")
            return False
    
    async def get_available_associates(self) -> List[AssociateInfo]:
        """
        Get list of available associates from the database.
        
        Returns:
            List[AssociateInfo]: List of available associates
        """
        try:
            logger.info("Getting available associates from database")
            # Get associates from the property database
            # This would typically query the associates table or extract from property data
            result = await self.supabase_service.get_unique_associates()
            logger.info(f"Found {len(result)} associates from database")
            
            associates = []
            for associate_data in result:
                associate = AssociateInfo(
                    id=associate_data.get('id', ''),
                    name=associate_data.get('name', ''),
                    email=associate_data.get('email', ''),
                    phone=associate_data.get('phone', ''),
                    specialization=associate_data.get('specialization', 'Real Estate'),
                    availability_hours=associate_data.get('availability_hours', '9:00 AM - 6:00 PM'),
                    timezone=associate_data.get('timezone', 'America/New_York')
                )
                associates.append(associate)
            
            return associates
            
        except Exception as e:
            logger.error(f"Error getting associates: {str(e)}")
            # Return default associates if database query fails
            logger.info("Falling back to default associates")
            return [
                AssociateInfo(
                    id="default-1",
                    name="Sarah Johnson",
                    email="sarah.johnson@example.com",
                    phone="(555) 123-4567",
                    specialization="Commercial Real Estate",
                    availability_hours="9:00 AM - 6:00 PM",
                    timezone="America/New_York"
                ),
                AssociateInfo(
                    id="default-2", 
                    name="Michael Chen",
                    email="michael.chen@example.com",
                    phone="(555) 234-5678",
                    specialization="Residential Real Estate",
                    availability_hours="8:00 AM - 7:00 PM",
                    timezone="America/New_York"
                )
            ]
    
    async def get_availability_slots(self, associate_id: str, days_ahead: int = 7) -> List[AvailabilitySlot]:
        """
        Get available time slots for an associate.
        
        Args:
            associate_id: ID of the associate
            days_ahead: Number of days to look ahead
            
        Returns:
            List[AvailabilitySlot]: Available time slots
        """
        try:
            # Get existing appointments for the associate
            existing_appointments = await self.supabase_service.get_associate_appointments(associate_id)
            
            # Generate availability slots
            slots = []
            now = datetime.now(timezone.utc)
            
            for day in range(1, days_ahead + 1):
                date = now + timedelta(days=day)
                
                # Skip weekends (assuming business hours)
                if date.weekday() >= 5:
                    continue
                
                # Generate time slots for business hours (9 AM - 6 PM)
                for hour in range(9, 18):
                    slot_time = date.replace(hour=hour, minute=0, second=0, microsecond=0)
                    
                    # Check if slot is already booked
                    is_booked = any(
                        abs((appointment['scheduled_time'] - slot_time).total_seconds()) < 3600
                        for appointment in existing_appointments
                    )
                    
                    if not is_booked:
                        slot = AvailabilitySlot(
                            datetime=slot_time,
                            associate_id=associate_id,
                            duration_minutes=60,
                            is_available=True
                        )
                        slots.append(slot)
            
            return slots[:10]  # Return first 10 available slots
            
        except Exception as e:
            logger.error(f"Error getting availability slots: {str(e)}")
            # Return sample slots if there's an error
            return self._generate_sample_slots(associate_id)
    
    def _generate_sample_slots(self, associate_id: str) -> List[AvailabilitySlot]:
        """Generate sample availability slots for demonstration."""
        slots = []
        now = datetime.now(timezone.utc)
        
        # Generate next 5 business days
        for day in range(1, 6):
            date = now + timedelta(days=day)
            if date.weekday() < 5:  # Monday to Friday
                # Morning slot
                morning_slot = AvailabilitySlot(
                    datetime=date.replace(hour=10, minute=0, second=0, microsecond=0),
                    associate_id=associate_id,
                    duration_minutes=60,
                    is_available=True
                )
                slots.append(morning_slot)
                
                # Afternoon slot
                afternoon_slot = AvailabilitySlot(
                    datetime=date.replace(hour=14, minute=0, second=0, microsecond=0),
                    associate_id=associate_id,
                    duration_minutes=60,
                    is_available=True
                )
                slots.append(afternoon_slot)
        
        return slots
    
    async def schedule_appointment(self, request: AppointmentRequest) -> AppointmentResponse:
        """
        Schedule an appointment with an associate.
        
        Args:
            request: Appointment request details
            
        Returns:
            AppointmentResponse: Appointment confirmation
        """
        try:
            # Validate the request
            if not request.associate_id or not request.scheduled_time:
                raise ValueError("Associate ID and scheduled time are required")

            # Standardise scheduled_time:
            #   • If the incoming time is naive, assume it is in the app's configured local_timezone.
            #   • Convert that value to UTC for conflict-checking & storage.
            from datetime import timezone
            from zoneinfo import ZoneInfo

            local_tz = ZoneInfo(getattr(settings, "local_timezone", "UTC"))

            if request.scheduled_time.tzinfo is None or request.scheduled_time.tzinfo.utcoffset(request.scheduled_time) is None:
                scheduled_local = request.scheduled_time.replace(tzinfo=local_tz)
            else:
                # already timezone-aware; keep as local for response
                scheduled_local = request.scheduled_time.astimezone(local_tz)

            scheduled_time_utc = scheduled_local.astimezone(timezone.utc)
 
            # Check if the slot is still available
            appointments = await self.supabase_service.get_associate_appointments(request.associate_id)
 
            # Check for conflicts
            for appointment in appointments:
                time_diff = abs((appointment['scheduled_time'] - scheduled_time_utc).total_seconds())
                if time_diff < 3600:  # Within 1 hour
                    return AppointmentResponse(
                        success=False,
                        appointment_id=None,
                        message="This time slot is no longer available. Please choose another time.",
                        scheduled_time=None,
                        associate_name=None
                    )
 
            # Create the appointment
            appointment_id = await self.supabase_service.create_appointment(
                session_id=request.session_id,
                associate_id=request.associate_id,
                user_name=request.user_name,
                user_email=request.user_email,
                user_phone=request.user_phone,
                scheduled_time=scheduled_time_utc,
                appointment_type=request.appointment_type,
                notes=request.notes
            )
            
            # Get associate info
            associates = await self.get_available_associates()
            associate_name = next(
                (a.name for a in associates if a.id == request.associate_id),
                "Associate"
            )
            
            # Try to add to calendar if configured
            calendar_event_id = None
            try:
                # Attempt to push to Google Calendar using OAuth credentials
                if settings.google_oauth_client_id and settings.google_oauth_client_secret and settings.google_calendar_id:
                    calendar_event_id = await self._add_to_calendar(request, associate_name)
            except Exception as e:
                logger.warning(f"Failed to add to calendar: {str(e)}")
            
            return AppointmentResponse(
                success=True,
                appointment_id=appointment_id,
                message=f"Appointment scheduled successfully with {associate_name}",
                scheduled_time=scheduled_local,
                associate_name=associate_name,
                calendar_event_id=calendar_event_id
            )
            
        except Exception as e:
            logger.error(f"Error scheduling appointment: {str(e)}")
            return AppointmentResponse(
                success=False,
                appointment_id=None,
                message=f"Failed to schedule appointment: {str(e)}",
                scheduled_time=None,
                associate_name=None
            )
    
    async def _add_to_calendar(self, request: AppointmentRequest, associate_name: str) -> Optional[str]:
        """Add appointment to Google Calendar using OAuth user credentials."""
        try:
            creds = self._get_oauth_creds()

            service = await self._run_sync(lambda: build("calendar", "v3", credentials=creds, cache_discovery=False))
             
            from zoneinfo import ZoneInfo
            user_tz = ZoneInfo(getattr(settings, "local_timezone", "UTC"))

            # Keep the *clock* time the user selected, but ensure it is tagged with the user_tz
            if request.scheduled_time.tzinfo is None or str(request.scheduled_time.tzinfo) in ("UTC", "UTC+00:00", "UTC-00:00"):
                start_local = request.scheduled_time.replace(tzinfo=user_tz)
            else:
                # Already timezone-aware but convert just in case
                start_local = request.scheduled_time.astimezone(user_tz)

            end_local = start_local + timedelta(hours=1)

            event = {
                "summary": f"Real Estate Consultation with {associate_name}",
                "description": f"Appointment with {request.user_name}\n\nNotes: {request.notes or 'No additional notes'}",
                "start": {
                    "dateTime": start_local.isoformat(),
                    "timeZone": str(user_tz),
                },
                "end": {
                    "dateTime": end_local.isoformat(),
                    "timeZone": str(user_tz),
                },
                "attendees": [
                    {"email": request.user_email},
                ],
                "reminders": {"useDefault": True},
            }

            def _insert():
                return service.events().insert(
                    calendarId=settings.google_calendar_id,
                    body=event,
                    sendUpdates="all"
                ).execute()

            created_event = await self._run_sync(_insert)
            return created_event.get("id")

        except HttpError as err:
            logger.error(f"Google Calendar API error: {err}")
            return None
        except Exception as e:
            logger.error(f"Error adding to calendar: {str(e)}")
            return None
    
    async def suggest_follow_up(self, session_id: str, days_ahead: int = 14) -> Optional[str]:
        """
        Suggest a follow-up appointment.
        
        Args:
            session_id: Current session ID
            days_ahead: Days ahead to suggest follow-up
            
        Returns:
            Optional[str]: Follow-up suggestion message
        """
        try:
            # Get available associates
            associates = await self.get_available_associates()
            if not associates:
                return None
            
            # Get availability for the first associate
            associate = associates[0]
            slots = await self.get_availability_slots(associate.id, days_ahead)
            
            if not slots:
                return None
            
            # Find a slot around 2 weeks from now
            target_date = datetime.now(timezone.utc) + timedelta(days=days_ahead)
            best_slot = min(slots, key=lambda x: abs((x.datetime - target_date).total_seconds()))
            
            # Format the suggestion
            formatted_time = best_slot.datetime.strftime('%A, %B %d at %I:%M %p')
            
            return f"I'll schedule a follow-up call with {associate.name} in 2 weeks to check on your property search. " \
                   f"They have availability on {formatted_time}. Would you like me to book this for you?"
            
        except Exception as e:
            logger.error(f"Error suggesting follow-up: {str(e)}")
            return None
    
    def generate_scheduling_offer(self, context: str = "") -> str:
        """
        Generate a smart scheduling offer based on context.
        
        Args:
            context: Context about the conversation
            
        Returns:
            str: Scheduling offer message
        """
        try:
            logger.info(f"Generating scheduling offer with context: '{context[:100]}...'")
            
            # Check if context already mentions inability to book directly
            if "cannot directly book" in context.lower() or "can't directly book" in context.lower():
                # Don't offer direct booking, offer to connect instead
                offers = [
                    "I can connect you with one of our associates who can help you directly. Would you like me to arrange a call?",
                    "Let me put you in touch with one of our specialists who can assist you further. Would you like to schedule a consultation?",
                    "Our team can help you with this directly. Would you like me to schedule a call with one of our associates?"
                ]
            else:
                # Standard scheduling offers
                offers = [
                    "I can arrange a meeting with one of our property experts. They have availability today at 3 PM or tomorrow at 10 AM.",
                    "Our associates are available to discuss your property needs in detail. Would you like to schedule a 30-minute call?",
                    "Would you like to speak with one of our associates about this? I can schedule a call at your convenience."
                ]
            
            # Add context-specific offers
            if "property" in context.lower():
                offers.append("Based on your interest in this property, would you like to schedule a viewing or consultation with our associate?")
            
            if "rent" in context.lower() or "lease" in context.lower():
                offers.append("Our leasing specialists can provide more details about rental terms. Would you like to schedule a call?")
            
            # Return a random offer for variety but avoid repetitive availability times
            import random
            selected_offer = random.choice(offers)
            logger.info(f"Generated scheduling offer: '{selected_offer}'")
            return selected_offer
            
        except Exception as e:
            logger.error(f"Error generating scheduling offer: {str(e)}")
            # Return a simple fallback offer
            fallback = "Would you like to schedule a meeting with one of our associates? I can help you book an appointment."
            logger.info(f"Using fallback offer: '{fallback}'")
            return fallback
    
    async def get_appointment_details(self, appointment_id: str) -> Optional[Dict[str, Any]]:
        """Get appointment details by ID."""
        try:
            return await self.supabase_service.get_appointment_details(appointment_id)
        except Exception as e:
            logger.error(f"Error getting appointment details: {str(e)}")
            return None
    
    async def cancel_appointment(self, appointment_id: str) -> bool:
        """Cancel an appointment."""
        try:
            return await self.supabase_service.cancel_appointment(appointment_id)
        except Exception as e:
            logger.error(f"Error canceling appointment: {str(e)}")
            return False
    
    async def reschedule_appointment(self, appointment_id: str, new_time: datetime) -> bool:
        """Reschedule an appointment to a new time."""
        try:
            return await self.supabase_service.reschedule_appointment(appointment_id, new_time)
        except Exception as e:
            logger.error(f"Error rescheduling appointment: {str(e)}")
            return False 

    def extract_associate_from_context(self, conversation_history: List[Dict]) -> Optional[str]:
        """
        Extract associate name mentioned in recent conversation context.
        
        Args:
            conversation_history: Recent conversation history
            
        Returns:
            str: Associate name if found, None otherwise
        """
        try:
            # Look for associate names in recent messages
            recent_messages = conversation_history[-5:]  # Check last 5 messages
            
            # Common patterns for associate mentions
            associate_patterns = [
                r'Associate \d+[,:]?\s*([A-Za-z\s]+)',
                r'associate[,:]?\s*([A-Za-z\s]+)',
                r'broker[,:]?\s*([A-Za-z\s]+)',
                r'contact[,:]?\s*([A-Za-z\s]+)',
                r'([A-Za-z]+\s+[A-Za-z]+)(?:\s*,|\s+and|\s*\))'  # Name pattern
            ]
            
            for msg in recent_messages:
                if msg.get('role') == 'assistant':
                    content = msg.get('content', '')
                    
                    # Try each pattern
                    for pattern in associate_patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        for match in matches:
                            name = match.strip()
                            # Filter out common words that aren't names
                            if name and len(name.split()) >= 2 and not any(word in name.lower() for word in [
                                'the', 'and', 'or', 'via', 'at', 'email', 'phone', 'contact', 'information',
                                'please', 'can', 'you', 'other', 'any', 'associates', 'include', 'specify'
                            ]):
                                logger.info(f"Found associate name from context: {name}")
                                return name
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting associate from context: {str(e)}")
            return None
    
    async def find_associate_by_name(self, name: str) -> Optional[str]:
        """
        Find associate ID by name from the database.
        
        Args:
            name: Associate name to search for
            
        Returns:
            str: Associate ID if found, None otherwise
        """
        try:
            # Strip common leading words (e.g. "with jack" -> "jack")
            sanitized = re.sub(r'^(with|associate|agent|broker)\s+', '', name, flags=re.IGNORECASE).strip()

            # Get all associates
            associates = await self.get_available_associates()
            
            # Try to match by name (case-insensitive)
            for associate in associates:
                if associate.name.lower() == sanitized.lower():
                    logger.info(f"Found exact match for associate: {sanitized} -> {associate.id}")
                    return associate.id
            
            # Try partial matching – substring or first/last-name matches
            for associate in associates:
                if sanitized.lower() in associate.name.lower() or associate.name.lower() in sanitized.lower():
                    logger.info(f"Found partial match for associate: {sanitized} -> {associate.name} ({associate.id})")
                    return associate.id

            # If user gave a single word, match it against first or last name tokens
            if len(sanitized.split()) == 1:
                for associate in associates:
                    tokens = associate.name.lower().split()
                    if sanitized.lower() in tokens:
                        logger.info(f"Matched single-word name '{sanitized}' to associate {associate.name} ({associate.id})")
                        return associate.id
            
            logger.info(f"No matching associate found for: {sanitized}")
            # Fuzzy match using difflib to handle small typos (e.g., "jack sprrow" -> "Jack Sparrow")
            associate_names = [a.name for a in associates]
            close_matches = difflib.get_close_matches(sanitized, associate_names, n=1, cutoff=0.7)
            if close_matches:
                best_match = close_matches[0]
                best_associate = next((a for a in associates if a.name == best_match), None)
                if best_associate:
                    logger.info(f"Using fuzzy match for associate: {name} -> {best_associate.name} ({best_associate.id})")
                    return best_associate.id

            logger.info(f"No matching associate found for: {name}")
            return None
            
        except Exception as e:
            logger.error(f"Error finding associate by name: {str(e)}")
            return None 

    def parse_associate_from_message(self, message: str) -> Optional[str]:
        """
        Parse associate name from the current user message.
        
        Args:
            message: Current user message
            
        Returns:
            str: Associate name if found, None otherwise
        """
        try:
            message_lower = message.lower()
            
            # Patterns for associate mentions in user messages (ordered by specificity)
            patterns = [
                # Two-word name patterns
                r'(?:associate|agent|broker)\s+([A-Za-z]+\s+[A-Za-z]+)',  # "associate Brenda Sparks"
                r'with\s+associate\s+([A-Za-z]+\s+[A-Za-z]+)',           # "with associate Brenda Sparks"
                r'with\s+([A-Za-z]+\s+[A-Za-z]+)',                       # "with cutler beckett"
                r'book.*?with\s+([A-Za-z]+\s+[A-Za-z]+)',                # "book with Brenda Sparks"
                r'schedule.*?with\s+([A-Za-z]+\s+[A-Za-z]+)',            # "schedule with Brenda Sparks"
                r'([A-Za-z]+\s+[A-Za-z]+)\s+tomorrow',                   # "Brenda Sparks tomorrow"
                r'([A-Za-z]+\s+[A-Za-z]+)\s+at\s+\d',                  # "Brenda Sparks at 10"
                # Single-word first-name patterns
                r'with\s+associate\s+([A-Za-z]+)',                      # "with associate Jack"
                r'with\s+([A-Za-z]+)',                                    # "with Jack"
                r'book.*?with\s+([A-Za-z]+)',                             # "book with Jack"
                r'schedule.*?with\s+([A-Za-z]+)',                         # "schedule with Jack"
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, message_lower, re.IGNORECASE)
                for match in matches:
                    name = match.strip()
                    # Filter out common words that aren't names
                    if name and not any(word in name.lower() for word in [
                        'the', 'and', 'or', 'via', 'at', 'email', 'phone', 'contact', 'information',
                        'please', 'can', 'you', 'other', 'any', 'associates', 'include', 'specify',
                        'tomorrow', 'today', 'appointment', 'meeting', 'schedule', 'book', 'call'
                    ]):
                        logger.info(f"Found associate name from user message: {name}")
                        return name
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing associate from message: {str(e)}")
            return None
    
    def parse_time_from_message(self, message: str) -> Optional[datetime]:
        """
        Parse specific time from the current user message.
        
        Args:
            message: Current user message
            
        Returns:
            datetime: Parsed time if found, None otherwise
        """
        try:
            message_lower = message.lower()
            now = datetime.now(timezone.utc)
            
            # Enhanced time patterns accepting a.m./p.m. variants (with optional dots)
            meridian = r'(?:a\.?m\.?|p\.?m\.?)'
            patterns = [
                (rf'tomorrow\s+at\s+(\d{{1,2}})\s*{meridian}', 1),
                (rf'tomorrow\s+(\d{{1,2}})\s*{meridian}', 1),
                (rf'today\s+at\s+(\d{{1,2}})\s*{meridian}', 0),
                (rf'at\s+(\d{{1,2}})\s*{meridian}', 1),
                (rf'(\d{{1,2}})\s*{meridian}', 1),
                (rf'tomorrow\s+at\s+(\d{{1,2}}):(\d{{2}})?\s*{meridian}', 1),
                (rf'tomorrow\s+(\d{{1,2}}):(\d{{2}})?\s*{meridian}', 1),
                (rf'at\s+(\d{{1,2}}):(\d{{2}})?\s*{meridian}', 1),
            ]
            
            for pattern, days_ahead in patterns:
                match = re.search(pattern, message_lower)
                if match:
                    groups = match.groups()
                    hour = int(groups[0])
                    # Determine minute (may be absent)
                    minute_group = groups[1] if len(groups) > 1 else None
                    minute = int(minute_group) if minute_group and minute_group.isdigit() else 0

                    # Detect am/pm by inspecting the matched substring
                    substring = match.group(0)
                    am_pm = 'pm' if re.search(r'p\.?m\.?', substring) else 'am'
                    
                    # Convert to 24-hour format
                    if am_pm == 'pm' and hour != 12:
                        hour += 12
                    elif am_pm == 'am' and hour == 12:
                        hour = 0
                    
                    # Calculate target date
                    target_date = now + timedelta(days=days_ahead)
                    scheduled_time = target_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    
                    logger.info(f"Parsed time from message: {scheduled_time}")
                    return scheduled_time
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing time from message: {str(e)}")
            return None 

    async def check_associate_availability(self, associate_id: str, scheduled_time: datetime) -> bool:
        """
        Check if an associate is available at the specified time.
        
        Args:
            associate_id: ID of the associate to check
            scheduled_time: Time to check availability for
            
        Returns:
            bool: True if associate is available, False otherwise
        """
        try:
            # Get existing appointments for this associate
            appointments = await self.supabase_service.get_associate_appointments(associate_id)
            
            # Check for conflicts within 1 hour
            for appointment in appointments:
                time_diff = abs((appointment['scheduled_time'] - scheduled_time).total_seconds())
                if time_diff < 3600:  # Within 1 hour
                    logger.info(f"Associate {associate_id} has conflict at {scheduled_time}")
                    return False
            
            logger.info(f"Associate {associate_id} is available at {scheduled_time}")
            return True
            
        except Exception as e:
            logger.error(f"Error checking associate availability: {str(e)}")
            return False
    
    async def find_available_associate_for_time(self, preferred_associates: List[str], scheduled_time: datetime) -> Optional[str]:
        """
        Find an available associate from a list of preferred associates.
        
        Args:
            preferred_associates: List of associate names to try
            scheduled_time: Time to check availability for
            
        Returns:
            str: Available associate ID, or None if none available
        """
        try:
            # Get all associates
            all_associates = await self.get_available_associates()
            
            # Try each preferred associate
            for preferred_name in preferred_associates:
                # Find associate by name
                associate_id = await self.find_associate_by_name(preferred_name)
                if associate_id:
                    # Check availability
                    if await self.check_associate_availability(associate_id, scheduled_time):
                        logger.info(f"Found available associate: {preferred_name} ({associate_id})")
                        return associate_id
                    else:
                        logger.info(f"Associate {preferred_name} is not available at {scheduled_time}")
            
            # If no preferred associates available, try any associate
            for associate in all_associates:
                if await self.check_associate_availability(associate.id, scheduled_time):
                    logger.info(f"Found alternative available associate: {associate.name} ({associate.id})")
                    return associate.id
            
            logger.info("No associates available at the requested time")
            return None
            
        except Exception as e:
            logger.error(f"Error finding available associate: {str(e)}")
            return None 

    def _names_match(self, a: str, b: str) -> bool:
        """Return True if two names are essentially the same (case-insensitive, substring or fuzzy >0.6)."""
        if not a or not b:
            return False
        a_lower, b_lower = a.lower(), b.lower()
        # Direct substring either direction
        if a_lower in b_lower or b_lower in a_lower:
            return True
        # Fuzzy similarity
        return difflib.SequenceMatcher(None, a_lower, b_lower).ratio() >= 0.6 