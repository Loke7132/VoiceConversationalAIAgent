from typing import List, Dict, Any, Optional, Tuple
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import base64
import requests
import io
import time

from ..config import settings

logger = logging.getLogger(__name__)


class GoogleCloudService:
    """Service for interacting with Google Cloud Speech-to-Text and Text-to-Speech APIs."""
    
    def __init__(self):
        self.service_account_path = settings.google_cloud_service_account_path
        self.project_id = settings.google_cloud_project_id
        self.executor = ThreadPoolExecutor(max_workers=5)
        self._access_token = None
        self._token_expiry = 0
        
        if not self.service_account_path or not self.project_id:
            raise ValueError("Google Cloud service account path and project ID must be configured")
        
        logger.info(f"Google Cloud service initialized for project: {self.project_id}")
    
    async def _get_access_token(self) -> str:
        """Get Google Cloud access token using service account credentials."""
        current_time = time.time()
        
        # Check if we have a valid token
        if self._access_token and current_time < self._token_expiry:
            return self._access_token
        
        def _get_token():
            try:
                # Read service account credentials
                with open(self.service_account_path, 'r') as f:
                    credentials = json.load(f)
                
                # Create JWT for token request
                import jwt
                from datetime import datetime, timedelta
                
                now = datetime.utcnow()
                payload = {
                    'iss': credentials['client_email'],
                    'scope': 'https://www.googleapis.com/auth/cloud-platform',
                    'aud': 'https://oauth2.googleapis.com/token',
                    'iat': now,
                    'exp': now + timedelta(hours=1)
                }
                
                # Sign JWT with private key
                private_key = credentials['private_key']
                token = jwt.encode(payload, private_key, algorithm='RS256')
                
                # Request access token
                response = requests.post(
                    'https://oauth2.googleapis.com/token',
                    data={
                        'grant_type': 'urn:ietf:params:oauth:grant-type:jwt-bearer',
                        'assertion': token
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    token_data = response.json()
                    access_token = token_data['access_token']
                    expires_in = token_data.get('expires_in', 3600)
                    
                    # Cache token with buffer time
                    self._access_token = access_token
                    self._token_expiry = current_time + expires_in - 300  # 5 min buffer
                    
                    return access_token
                else:
                    response.raise_for_status()
                    
            except Exception as e:
                logger.error(f"Failed to get Google Cloud access token: {e}")
                raise e
        
        return await asyncio.get_event_loop().run_in_executor(self.executor, _get_token)
    
    async def transcribe_audio(
        self, 
        audio_data: bytes, 
        max_retries: int = 3,
        language_code: str = "en-US"
    ) -> str:
        """
        Transcribe audio using Google Cloud Speech-to-Text.
        
        Args:
            audio_data: Audio data as bytes
            max_retries: Maximum number of retries
            
        Returns:
            Transcribed text
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                access_token = await self._get_access_token()
                
                def _transcribe():
                    try:
                        # Encode audio data to base64
                        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                        
                        # Prepare request payload
                        payload = {
                            "config": {
                                "encoding": "WEBM_OPUS",  # Adjust based on your audio format
                                "sampleRateHertz": 48000,  # Adjust based on your audio format
                                "languageCode": language_code,
                                "enableAutomaticPunctuation": True,
                                "enableWordTimeOffsets": False,
                                "model": "latest_long"  # Use latest long model for best accuracy
                            },
                            "audio": {
                                "content": audio_base64
                            }
                        }
                        
                        headers = {
                            'Authorization': f'Bearer {access_token}',
                            'Content-Type': 'application/json'
                        }
                        
                        # Make API call
                        response = requests.post(
                            'https://speech.googleapis.com/v1/speech:recognize',
                            headers=headers,
                            json=payload,
                            timeout=60
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            # Extract transcript from response
                            if 'results' in result and result['results']:
                                alternatives = result['results'][0].get('alternatives', [])
                                if alternatives:
                                    transcript = alternatives[0].get('transcript', '')
                                    logger.info(f"Google Cloud STT transcription: {transcript[:100]}...")
                                    return transcript
                            
                            logger.warning("No transcription results returned")
                            return ""
                        else:
                            response.raise_for_status()
                            
                    except Exception as e:
                        logger.error(f"Google Cloud STT API error: {e}")
                        raise e
                
                result = await asyncio.get_event_loop().run_in_executor(
                    self.executor, _transcribe
                )
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(f"Google Cloud STT attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
        
        # If we get here, all attempts failed
        logger.error(f"Google Cloud STT failed after {max_retries} attempts. Last error: {last_error}")
        raise Exception(f"Google Cloud STT failed after {max_retries} attempts. Last error: {last_error}")
    
    async def transcribe_audio_with_language_detection(
        self, 
        audio_data: bytes, 
        max_retries: int = 3
    ) -> Tuple[str, str]:
        """
        Transcribe audio with automatic language detection.
        
        Args:
            audio_data: Audio data as bytes
            max_retries: Maximum number of retries
            
        Returns:
            Tuple of (transcribed_text, detected_language_code)
        """
        # Language codes to try in order of preference - including Tamil and other Indian languages
        language_codes = [
            "en-US", "es-ES", "fr-FR", "de-DE", "it-IT", "pt-BR", "ru-RU", 
            "zh-CN", "ja-JP", "ko-KR", "ar-SA", "hi-IN", "ta-IN", "te-IN", 
            "bn-IN", "mr-IN", "gu-IN", "kn-IN", "ml-IN", "pa-IN", "ur-IN",
            "nl-NL", "sv-SE", "no-NO", "da-DK", "fi-FI", "pl-PL", "cs-CZ", 
            "hu-HU", "tr-TR", "th-TH", "vi-VN", "id-ID", "ms-MY", "fil-PH"
        ]
        
        last_error = None
        
        for attempt in range(max_retries):
            try:
                access_token = await self._get_access_token()
                
                def _transcribe_with_detection():
                    try:
                        # Encode audio data to base64
                        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                        
                        # Prepare request payload with multiple language codes
                        payload = {
                            "config": {
                                "encoding": "WEBM_OPUS",  # Adjust based on your audio format
                                "sampleRateHertz": 48000,  # Adjust based on your audio format
                                "languageCode": "en-US",  # Primary language
                                "alternativeLanguageCodes": language_codes[1:15],  # More languages including Tamil
                                "enableAutomaticPunctuation": True,
                                "enableWordTimeOffsets": False,
                                "model": "latest_long"  # Use latest long model for best accuracy
                            },
                            "audio": {
                                "content": audio_base64
                            }
                        }
                        
                        headers = {
                            'Authorization': f'Bearer {access_token}',
                            'Content-Type': 'application/json'
                        }
                        
                        # Make API call
                        response = requests.post(
                            'https://speech.googleapis.com/v1/speech:recognize',
                            headers=headers,
                            json=payload,
                            timeout=60
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            # Extract transcript and language from response
                            if 'results' in result and result['results']:
                                alternatives = result['results'][0].get('alternatives', [])
                                if alternatives:
                                    transcript = alternatives[0].get('transcript', '')
                                    
                                    # Extract detected language
                                    detected_language = result['results'][0].get('languageCode', 'en-US')
                                    
                                    logger.info(f"Google Cloud STT transcription with language detection: {transcript[:100]}... (Language: {detected_language})")
                                    return transcript, detected_language
                            
                            logger.warning("No transcription results returned")
                            return "", "en-US"
                        else:
                            response.raise_for_status()
                            
                    except Exception as e:
                        logger.error(f"Google Cloud STT with language detection API error: {e}")
                        raise e
                
                result = await asyncio.get_event_loop().run_in_executor(
                    self.executor, _transcribe_with_detection
                )
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(f"Google Cloud STT with language detection attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
        
        # If we get here, all attempts failed
        logger.error(f"Google Cloud STT with language detection failed after {max_retries} attempts. Last error: {last_error}")
        raise Exception(f"Google Cloud STT with language detection failed after {max_retries} attempts. Last error: {last_error}")
    
    async def transcribe_audio_multilingual(
        self, 
        audio_data: bytes, 
        max_retries: int = 3
    ) -> Tuple[str, str]:
        """
        Transcribe audio with better multilingual support.
        
        This method tries multiple language codes sequentially until it gets a good result.
        
        Args:
            audio_data: Audio data as bytes
            max_retries: Maximum number of retries per language
            
        Returns:
            Tuple of (transcribed_text, detected_language_code)
        """
        # Comprehensive language codes to try, including Tamil and other Indian languages
        language_codes = [
            "en-US", "hi-IN", "ta-IN", "te-IN", "bn-IN", "mr-IN", "gu-IN", 
            "kn-IN", "ml-IN", "pa-IN", "ur-IN", "es-ES", "fr-FR", "de-DE", 
            "it-IT", "pt-BR", "ru-RU", "zh-CN", "ja-JP", "ko-KR", "ar-SA",
            "th-TH", "vi-VN", "id-ID", "ms-MY", "fil-PH", "nl-NL", "sv-SE", 
            "no-NO", "da-DK", "fi-FI", "pl-PL", "cs-CZ", "hu-HU", "tr-TR"
        ]
        
        best_result = ("", "en-US")
        best_confidence = 0.0
        
        for language_code in language_codes:
            try:
                # Try transcribing with this specific language
                transcription = await self.transcribe_audio(audio_data, max_retries=1, language_code=language_code)
                
                if transcription and len(transcription.strip()) > 0:
                    # Simple heuristic: longer transcription with actual words is likely better
                    confidence = len(transcription.strip().split())
                    
                    logger.info(f"Language {language_code}: '{transcription[:50]}...' (confidence: {confidence})")
                    
                    if confidence > best_confidence:
                        best_result = (transcription, language_code)
                        best_confidence = confidence
                        
                        # If we get a good result (5+ words), we can stop early
                        if confidence >= 5:
                            logger.info(f"Good transcription found with {language_code}: {transcription[:100]}...")
                            break
                            
            except Exception as e:
                logger.debug(f"Language {language_code} failed: {e}")
                continue
        
        if best_result[0]:
            logger.info(f"Best transcription: {best_result[0][:100]}... (Language: {best_result[1]})")
            return best_result
        else:
            # Fallback to English if nothing worked
            logger.warning("No successful transcription found, falling back to English")
            return await self.transcribe_audio(audio_data, max_retries=max_retries, language_code="en-US"), "en-US"
    
    async def text_to_speech(
        self, 
        text: str, 
        voice_id: str = "en-US-Standard-A",  # Default Google Cloud voice
        max_retries: int = 3
    ) -> bytes:
        """
        Convert text to speech using Google Cloud Text-to-Speech.
        
        Args:
            text: Text to convert to speech
            voice_id: Voice identifier (Google Cloud voice name)
            max_retries: Maximum number of retries
            
        Returns:
            Audio data as bytes
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                access_token = await self._get_access_token()
                
                def _text_to_speech():
                    try:
                        # Parse voice_id to extract language and voice name
                        voice_parts = voice_id.split('-')
                        if len(voice_parts) >= 3:
                            language_code = f"{voice_parts[0]}-{voice_parts[1]}"  # e.g., "en-US"
                            voice_name = voice_id  # Full voice name
                        else:
                            language_code = "en-US"
                            voice_name = "en-US-Standard-A"
                        
                        # Prepare request payload
                        payload = {
                            "input": {
                                "text": text
                            },
                            "voice": {
                                "languageCode": language_code,
                                "name": voice_name,
                                "ssmlGender": "NEUTRAL"
                            },
                            "audioConfig": {
                                "audioEncoding": "MP3",
                                "speakingRate": 1.0,
                                "pitch": 0.0,
                                "volumeGainDb": 0.0
                            }
                        }
                        
                        headers = {
                            'Authorization': f'Bearer {access_token}',
                            'Content-Type': 'application/json'
                        }
                        
                        # Make API call
                        response = requests.post(
                            'https://texttospeech.googleapis.com/v1/text:synthesize',
                            headers=headers,
                            json=payload,
                            timeout=60
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            # Extract and decode audio content
                            if 'audioContent' in result:
                                audio_base64 = result['audioContent']
                                audio_data = base64.b64decode(audio_base64)
                                logger.info(f"Google Cloud TTS generated {len(audio_data)} bytes of audio")
                                return audio_data
                            else:
                                raise Exception("No audioContent in response")
                        else:
                            response.raise_for_status()
                            
                    except Exception as e:
                        logger.error(f"Google Cloud TTS API error: {e}")
                        raise e
                
                result = await asyncio.get_event_loop().run_in_executor(
                    self.executor, _text_to_speech
                )
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(f"Google Cloud TTS attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
        
        # If we get here, all attempts failed
        logger.error(f"Google Cloud TTS failed after {max_retries} attempts. Last error: {last_error}")
        raise Exception(f"Google Cloud TTS failed after {max_retries} attempts. Last error: {last_error}")
    
    async def get_available_voices(self) -> List[Dict[str, Any]]:
        """
        Get available voices from Google Cloud Text-to-Speech.
        
        Returns:
            List of available voices
        """
        try:
            access_token = await self._get_access_token()
            
            def _get_voices():
                headers = {
                    'Authorization': f'Bearer {access_token}',
                    'Content-Type': 'application/json'
                }
                
                response = requests.get(
                    'https://texttospeech.googleapis.com/v1/voices',
                    headers=headers,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    voices = []
                    
                    for voice in result.get('voices', []):
                        voices.append({
                            'voice_id': voice.get('name', ''),
                            'name': voice.get('name', ''),
                            'language_codes': voice.get('languageCodes', []),
                            'ssml_gender': voice.get('ssmlGender', ''),
                            'natural_sample_rate_hertz': voice.get('naturalSampleRateHertz', 0)
                        })
                    
                    return voices
                else:
                    response.raise_for_status()
            
            return await asyncio.get_event_loop().run_in_executor(self.executor, _get_voices)
            
        except Exception as e:
            logger.error(f"Failed to get Google Cloud voices: {e}")
            return []
    
    async def health_check(self) -> bool:
        """
        Check if Google Cloud services are accessible.
        
        Returns:
            True if services are healthy, False otherwise
        """
        try:
            access_token = await self._get_access_token()
            
            def _health_check():
                headers = {
                    'Authorization': f'Bearer {access_token}',
                    'Content-Type': 'application/json'
                }
                
                # Simple API call to check connectivity
                response = requests.get(
                    'https://texttospeech.googleapis.com/v1/voices?languageCode=en-US',
                    headers=headers,
                    timeout=10
                )
                
                return response.status_code == 200
            
            return await asyncio.get_event_loop().run_in_executor(self.executor, _health_check)
            
        except Exception as e:
            logger.error(f"Google Cloud health check failed: {e}")
            return False
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get information about the Google Cloud service."""
        return {
            "service_name": "Google Cloud Speech & Text-to-Speech",
            "project_id": self.project_id,
            "stt_model": "latest_long",
            "tts_format": "MP3",
            "languages_supported": "100+",
            "api_version": "v1"
        }
    
    async def get_subscription_info(self) -> Dict[str, Any]:
        """Get Google Cloud project information."""
        try:
            access_token = await self._get_access_token()
            
            def _get_project_info():
                headers = {
                    'Authorization': f'Bearer {access_token}',
                    'Content-Type': 'application/json'
                }
                
                # Get project information
                response = requests.get(
                    f'https://cloudresourcemanager.googleapis.com/v1/projects/{self.project_id}',
                    headers=headers,
                    timeout=30
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    return {"project_id": self.project_id, "status": "active"}
            
            return await asyncio.get_event_loop().run_in_executor(self.executor, _get_project_info)
            
        except Exception as e:
            logger.error(f"Failed to get Google Cloud project info: {e}")
            return {"project_id": self.project_id, "status": "unknown"}
    
    def estimate_cost(self, text: str) -> Dict[str, Any]:
        """Estimate cost for Google Cloud TTS request."""
        character_count = len(text)
        
        # Google Cloud TTS pricing (approximate, as of 2024)
        # Standard voices: $4.00 per 1 million characters
        # WaveNet voices: $16.00 per 1 million characters
        
        standard_cost = (character_count / 1_000_000) * 4.00
        wavenet_cost = (character_count / 1_000_000) * 16.00
        
        return {
            "character_count": character_count,
            "estimated_cost_standard_usd": round(standard_cost, 6),
            "estimated_cost_wavenet_usd": round(wavenet_cost, 6),
            "currency": "USD",
            "pricing_date": "2024"
        } 

    async def get_language_voice_mappings(self) -> Dict[str, Dict[str, str]]:
        """
        Get dynamic language and voice mappings from available Google Cloud voices.
        
        Returns:
            Dictionary with language mappings and voice mappings
        """
        try:
            voices = await self.get_available_voices()
            
            # Build language mappings (STT format to ISO 639-1)
            language_mappings = {}
            voice_mappings = {}
            
            # Process each voice to build mappings
            for voice in voices:
                voice_name = voice.get('name', '')
                language_codes = voice.get('language_codes', [])
                
                for lang_code in language_codes:
                    # Convert from Google Cloud format (e.g., "en-US") to ISO format (e.g., "en")
                    iso_code = lang_code.split('-')[0] if '-' in lang_code else lang_code
                    
                    # Build language mapping (STT format to ISO format)
                    language_mappings[lang_code] = iso_code
                    
                    # Build voice mapping (ISO format to voice name)
                    # Prefer Standard voices, then Wavenet, then Neural
                    if iso_code not in voice_mappings or 'Standard' in voice_name:
                        voice_mappings[iso_code] = voice_name
                    elif 'Wavenet' in voice_name and 'Standard' not in voice_mappings.get(iso_code, ''):
                        voice_mappings[iso_code] = voice_name
                    elif 'Neural' in voice_name and 'Standard' not in voice_mappings.get(iso_code, '') and 'Wavenet' not in voice_mappings.get(iso_code, ''):
                        voice_mappings[iso_code] = voice_name
            
            # Add common fallbacks if not found
            fallback_mappings = {
                "en-US": "en", "es-ES": "es", "fr-FR": "fr", "de-DE": "de", "it-IT": "it",
                "pt-BR": "pt", "ru-RU": "ru", "zh-CN": "zh", "ja-JP": "ja", "ko-KR": "ko",
                "ar-SA": "ar", "hi-IN": "hi", "ta-IN": "ta", "te-IN": "te", "bn-IN": "bn",
                "mr-IN": "mr", "gu-IN": "gu", "kn-IN": "kn", "ml-IN": "ml", "pa-IN": "pa",
                "ur-IN": "ur", "nl-NL": "nl", "sv-SE": "sv", "no-NO": "no", "da-DK": "da", 
                "fi-FI": "fi", "pl-PL": "pl", "cs-CZ": "cs", "hu-HU": "hu", "tr-TR": "tr",
                "th-TH": "th", "vi-VN": "vi", "id-ID": "id", "ms-MY": "ms", "fil-PH": "tl"
            }
            
            # Add fallbacks only if not already present
            for stt_code, iso_code in fallback_mappings.items():
                if stt_code not in language_mappings:
                    language_mappings[stt_code] = iso_code
                if iso_code not in voice_mappings:
                    voice_mappings[iso_code] = f"{stt_code}-Standard-A"
            
            logger.info(f"Built dynamic mappings: {len(language_mappings)} language mappings, {len(voice_mappings)} voice mappings")
            
            return {
                "language_mappings": language_mappings,
                "voice_mappings": voice_mappings,
                "reverse_language_mappings": {v: k for k, v in language_mappings.items()}
            }
            
        except Exception as e:
            logger.error(f"Failed to build dynamic mappings: {e}")
            # Return minimal fallback mappings
            return {
                "language_mappings": {"en-US": "en", "es-ES": "es", "fr-FR": "fr", "de-DE": "de", "hi-IN": "hi", "ta-IN": "ta"},
                "voice_mappings": {"en": "en-US-Standard-A", "es": "es-ES-Standard-A", "fr": "fr-FR-Standard-A", "de": "de-DE-Standard-A", "hi": "hi-IN-Standard-A", "ta": "ta-IN-Standard-A"},
                "reverse_language_mappings": {"en": "en-US", "es": "es-ES", "fr": "fr-FR", "de": "de-DE", "hi": "hi-IN", "ta": "ta-IN"}
            }

    async def get_best_voice_for_language(self, language_code: str) -> str:
        """
        Get the best available voice for a given language code.
        
        Args:
            language_code: Language code (e.g., "en", "es", "ta")
            
        Returns:
            Best voice name for the language
        """
        try:
            mappings = await self.get_language_voice_mappings()
            voice_mappings = mappings.get("voice_mappings", {})
            
            # Try to find exact match
            if language_code in voice_mappings:
                return voice_mappings[language_code]
            
            # Try to find by prefix (e.g., "en" matches "en-US")
            for lang, voice in voice_mappings.items():
                if lang.startswith(language_code) or language_code.startswith(lang):
                    return voice
            
            # Default to English if no match found
            return voice_mappings.get("en", "en-US-Standard-A")
            
        except Exception as e:
            logger.error(f"Failed to get best voice for {language_code}: {e}")
            return "en-US-Standard-A"

    async def convert_language_code(self, from_format: str, to_format: str, language_code: str) -> str:
        """
        Convert language code between different formats.
        
        Args:
            from_format: Source format ("stt", "iso", "tts")
            to_format: Target format ("stt", "iso", "tts")
            language_code: Language code to convert
            
        Returns:
            Converted language code
        """
        try:
            mappings = await self.get_language_voice_mappings()
            language_mappings = mappings.get("language_mappings", {})
            reverse_mappings = mappings.get("reverse_language_mappings", {})
            
            if from_format == "stt" and to_format == "iso":
                return language_mappings.get(language_code, language_code.split('-')[0])
            elif from_format == "iso" and to_format == "stt":
                return reverse_mappings.get(language_code, f"{language_code}-US" if language_code == "en" else f"{language_code}-{language_code.upper()}")
            elif from_format == "iso" and to_format == "tts":
                return reverse_mappings.get(language_code, f"{language_code}-US" if language_code == "en" else f"{language_code}-{language_code.upper()}")
            else:
                return language_code
                
        except Exception as e:
            logger.error(f"Failed to convert language code {language_code} from {from_format} to {to_format}: {e}")
            return language_code 