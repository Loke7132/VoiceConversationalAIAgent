from typing import Dict, Any, Optional, Tuple
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import requests
import time

from ..config import settings

logger = logging.getLogger(__name__)


class TranslationService:
    """Service for language detection and translation using Google Translate API."""
    
    def __init__(self):
        self.service_account_path = settings.google_cloud_service_account_path
        self.project_id = settings.google_cloud_project_id
        self.executor = ThreadPoolExecutor(max_workers=5)
        self._access_token = None
        self._token_expiry = 0
        
        if not self.service_account_path or not self.project_id:
            raise ValueError("Google Cloud service account path and project ID must be configured")
        
        logger.info(f"Translation service initialized for project: {self.project_id}")
        
        # Check if we can get API key from Gemini settings as fallback
        try:
            self.api_key = settings.get_gemini_api_keys()[0] if hasattr(settings, 'get_gemini_api_keys') else None
        except:
            self.api_key = None
    
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
    
    async def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect the language of the given text using Google Translate API v2.
        
        Args:
            text: Text to detect language for
            
        Returns:
            Tuple of (language_code, confidence)
        """
        try:
            access_token = await self._get_access_token()
            
            def _detect():
                headers = {
                    'Authorization': f'Bearer {access_token}',
                    'Content-Type': 'application/json'
                }
                
                # Use v2 API directly since it works
                v2_url = 'https://translation.googleapis.com/language/translate/v2/detect'
                v2_payload = {'q': text}
                
                response = requests.post(
                    v2_url,
                    headers=headers,
                    json=v2_payload,
                    timeout=30
                )
                
                logger.info(f"Language detection response status: {response.status_code}")
                logger.debug(f"Language detection response: {response.text[:200]}...")
                
                if response.status_code == 200:
                    result = response.json()
                    if 'data' in result and 'detections' in result['data']:
                        detections = result['data']['detections'][0]
                        if detections:
                            language_code = detections[0].get('language', 'en')
                            confidence = detections[0].get('confidence', 0.0)
                            logger.info(f"Detected language: {language_code} (confidence: {confidence})")
                            return language_code, confidence
                
                # Default to English if detection fails
                logger.warning("Language detection failed, defaulting to English")
                return 'en', 1.0
                
            return await asyncio.get_event_loop().run_in_executor(self.executor, _detect)
            
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            # Default to English if detection fails
            return 'en', 1.0
    
    async def translate_text(
        self, 
        text: str, 
        target_language: str, 
        source_language: Optional[str] = None
    ) -> str:
        """
        Translate text from source language to target language using Google Translate API v2.
        
        Args:
            text: Text to translate
            target_language: Target language code (e.g., 'en', 'es', 'fr')
            source_language: Source language code (optional, will auto-detect if None)
            
        Returns:
            Translated text
        """
        try:
            # If source and target are the same, return original text
            if source_language == target_language:
                logger.info(f"Source and target languages are the same ({source_language}), returning original text")
                return text
            
            access_token = await self._get_access_token()
            
            def _translate():
                headers = {
                    'Authorization': f'Bearer {access_token}',
                    'Content-Type': 'application/json'
                }
                
                logger.info(f"Translating from {source_language or 'auto'} to {target_language}: {text[:50]}...")
                
                # Use v2 API directly since it works
                v2_url = 'https://translation.googleapis.com/language/translate/v2'
                
                v2_payload = {
                    'q': text,
                    'target': target_language,
                    'format': 'text'
                }
                
                if source_language:
                    v2_payload['source'] = source_language
                
                response = requests.post(
                    v2_url,
                    headers=headers,
                    json=v2_payload,
                    timeout=30
                )
                
                logger.info(f"Translation response status: {response.status_code}")
                logger.debug(f"Translation response: {response.text[:200]}...")
                
                if response.status_code == 200:
                    result = response.json()
                    if 'data' in result and 'translations' in result['data']:
                        translations = result['data']['translations']
                        if translations:
                            translated_text = translations[0]['translatedText']
                            logger.info(f"Translation successful: {translated_text[:50]}...")
                            return translated_text
                
                # If translation fails, log the error and return original text
                logger.error(f"Translation failed. Status: {response.status_code}, Response: {response.text[:200]}...")
                return text
                
            return await asyncio.get_event_loop().run_in_executor(self.executor, _translate)
            
        except Exception as e:
            logger.error(f"Translation failed with exception: {e}")
            import traceback
            logger.error(f"Translation traceback: {traceback.format_exc()}")
            # Return original text if translation fails
            return text
    
    async def translate_to_english(self, text: str) -> Tuple[str, str]:
        """
        Detect language and translate to English if needed.
        
        Args:
            text: Text to translate
            
        Returns:
            Tuple of (translated_text, detected_language)
        """
        try:
            # Detect language
            detected_language, confidence = await self.detect_language(text)
            
            logger.info(f"Detected language: {detected_language} (confidence: {confidence:.2f})")
            
            # If already English, return as is
            if detected_language == 'en':
                return text, detected_language
            
            # Translate to English
            translated_text = await self.translate_text(text, 'en', detected_language)
            
            logger.info(f"Translated from {detected_language} to English: {text[:50]}... -> {translated_text[:50]}...")
            
            return translated_text, detected_language
            
        except Exception as e:
            logger.error(f"Translation to English failed: {e}")
            return text, 'en'
    
    async def translate_from_english(self, text: str, target_language: str) -> str:
        """
        Translate from English to target language.
        
        Args:
            text: English text to translate
            target_language: Target language code
            
        Returns:
            Translated text
        """
        try:
            # If target is English, return as is
            if target_language == 'en':
                return text
            
            translated_text = await self.translate_text(text, target_language, 'en')
            
            logger.info(f"Translated from English to {target_language}: {text[:50]}... -> {translated_text[:50]}...")
            
            return translated_text
            
        except Exception as e:
            logger.error(f"Translation from English failed: {e}")
            return text
    
    def get_supported_languages(self) -> Dict[str, str]:
        """
        Get supported languages for translation.
        
        Returns:
            Dictionary of language codes and names
        """
        return {
            'af': 'Afrikaans',
            'ar': 'Arabic',
            'bg': 'Bulgarian',
            'bn': 'Bengali',
            'ca': 'Catalan',
            'cs': 'Czech',
            'da': 'Danish',
            'de': 'German',
            'el': 'Greek',
            'en': 'English',
            'es': 'Spanish',
            'et': 'Estonian',
            'fa': 'Persian',
            'fi': 'Finnish',
            'fr': 'French',
            'gu': 'Gujarati',
            'he': 'Hebrew',
            'hi': 'Hindi',
            'hr': 'Croatian',
            'hu': 'Hungarian',
            'id': 'Indonesian',
            'it': 'Italian',
            'ja': 'Japanese',
            'kn': 'Kannada',
            'ko': 'Korean',
            'lt': 'Lithuanian',
            'lv': 'Latvian',
            'mk': 'Macedonian',
            'ml': 'Malayalam',
            'mr': 'Marathi',
            'nl': 'Dutch',
            'no': 'Norwegian',
            'pl': 'Polish',
            'pt': 'Portuguese',
            'ro': 'Romanian',
            'ru': 'Russian',
            'sk': 'Slovak',
            'sl': 'Slovenian',
            'so': 'Somali',
            'sq': 'Albanian',
            'sv': 'Swedish',
            'sw': 'Swahili',
            'ta': 'Tamil',
            'te': 'Telugu',
            'th': 'Thai',
            'tl': 'Filipino',
            'tr': 'Turkish',
            'uk': 'Ukrainian',
            'ur': 'Urdu',
            'vi': 'Vietnamese',
            'zh': 'Chinese (Simplified)',
            'zh-TW': 'Chinese (Traditional)'
        }
    
    async def get_supported_languages_dynamic(self) -> Dict[str, str]:
        """
        Get supported languages dynamically from Google Translate API.
        
        Returns:
            Dictionary of language codes and names
        """
        try:
            access_token = await self._get_access_token()
            
            def _get_languages():
                headers = {
                    'Authorization': f'Bearer {access_token}',
                    'Content-Type': 'application/json'
                }
                
                # Use v2 API to get supported languages
                response = requests.get(
                    'https://translation.googleapis.com/language/translate/v2/languages',
                    headers=headers,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    languages = {}
                    
                    if 'data' in result and 'languages' in result['data']:
                        for lang in result['data']['languages']:
                            code = lang.get('language', '')
                            name = lang.get('name', code)
                            if code:
                                languages[code] = name
                    
                    logger.info(f"Fetched {len(languages)} supported languages from Google Translate API")
                    return languages
                else:
                    logger.warning(f"Failed to fetch languages from API: {response.status_code}")
                    return self.get_supported_languages()  # Fallback to static list
                    
            return await asyncio.get_event_loop().run_in_executor(self.executor, _get_languages)
            
        except Exception as e:
            logger.error(f"Failed to get supported languages dynamically: {e}")
            return self.get_supported_languages()  # Fallback to static list
    
    async def health_check(self) -> bool:
        """
        Check if translation service is accessible.
        
        Returns:
            True if service is healthy, False otherwise
        """
        try:
            # Test with a simple translation
            test_text = "Hello, world!"
            translated = await self.translate_text(test_text, 'es', 'en')
            return translated != test_text  # Should be different if translation worked
            
        except Exception as e:
            logger.error(f"Translation health check failed: {e}")
            return False
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get information about the translation service."""
        return {
            "service_name": "Google Translate",
            "project_id": self.project_id,
            "supported_languages": len(self.get_supported_languages()),
            "api_version": "v2"
        } 