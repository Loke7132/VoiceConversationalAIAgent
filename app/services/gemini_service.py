from typing import List, Dict, Any, Optional
import logging
import asyncio
import time
import requests
import json
from concurrent.futures import ThreadPoolExecutor
import random

from ..config import settings

logger = logging.getLogger(__name__)


class GeminiService:
    """Service for interacting with Google Gemini API."""
    
    def __init__(self):
        self.api_keys = settings.get_gemini_api_keys()
        self.model = settings.gemini_model
        # Fallback models in order of preference
        self.fallback_models = [
            "gemini-1.5-pro-latest",
            "gemini-1.5-pro",
            "gemini-1.5-flash-latest", 
            "gemini-1.5-flash",
            "gemini-pro"
        ]
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        self.executor = ThreadPoolExecutor(max_workers=5)
        self._current_key_index = 0
        self._failed_keys = set()
        self._failed_models = set()
        
        if not self.api_keys:
            raise ValueError("No Gemini API keys configured")
        
        logger.info(f"Gemini service initialized with {len(self.api_keys)} API keys")
    
    def _get_next_client_key(self) -> Optional[str]:
        """Get the next available API key."""
        if len(self._failed_keys) >= len(self.api_keys):
            # Reset failed keys if all have failed
            self._failed_keys.clear()
            logger.warning("All Gemini API keys failed, resetting failed keys set")
        
        # Find an API key that hasn't recently failed
        attempts = 0
        while attempts < len(self.api_keys):
            current_key = self.api_keys[self._current_key_index]
            self._current_key_index = (self._current_key_index + 1) % len(self.api_keys)
            
            if current_key not in self._failed_keys:
                return current_key
            
            attempts += 1
        
        # If all keys have failed, return a random one anyway
        if self.api_keys:
            return random.choice(self.api_keys)
        
        return None
    
    def _handle_api_error(self, error: Exception, api_key: str):
        """Handle API errors and mark keys as failed if necessary."""
        error_str = str(error).lower()
        
        # Check for rate limiting or auth errors
        if any(keyword in error_str for keyword in ['rate limit', '429', 'quota', 'unauthorized', '401', '403']):
            logger.warning(f"Marking Gemini API key as failed due to: {error_str}")
            self._failed_keys.add(api_key)
        else:
            logger.error(f"Gemini API error: {error_str}")
    
    async def _run_sync(self, func, *args, **kwargs):
        """Run synchronous function in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args, **kwargs)
    
    async def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: int = 1000,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_retries: int = 3
    ) -> str:
        """
        Generate a response using Gemini API.
        
        Args:
            messages: List of messages in OpenAI format
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_retries: Maximum number of retries
            
        Returns:
            Generated response text
        """
        last_error = None
        
        for attempt in range(max_retries):
            api_key = self._get_next_client_key()
            if not api_key:
                raise Exception("No available Gemini API keys")
            
            try:
                def _generate():
                    # Convert OpenAI format messages to Gemini format
                    # Gemini requires alternating user/model messages
                    contents = []
                    system_message = ""
                    
                    # Extract system message first
                    for msg in messages:
                        if msg["role"] == "system":
                            system_message = msg["content"] + "\n\n"
                            break
                    
                    # Process conversation messages and ensure proper alternation
                    last_role = None
                    current_user_text = ""
                    current_model_text = ""
                    
                    for msg in messages:
                        role = msg["role"]
                        content = msg["content"]
                        
                        # Skip empty content
                        if not content or not content.strip():
                            continue
                            
                        if role == "system":
                            continue  # Already handled above
                        elif role == "user":
                            # If we're switching from model to user, save the model message first
                            if last_role == "model" and current_model_text:
                                contents.append({
                                    "role": "model",
                                    "parts": [{"text": current_model_text.strip()}]
                                })
                                current_model_text = ""
                            
                            # Accumulate user content
                            current_user_text += (" " if current_user_text else "") + content
                            last_role = "user"
                            
                        elif role == "assistant":
                            # If we're switching from user to model, save the user message first
                            if last_role == "user" and current_user_text:
                                # Add system message to first user message
                                text_to_add = current_user_text
                                if system_message and len(contents) == 0:
                                    text_to_add = system_message + text_to_add
                                
                                contents.append({
                                    "role": "user", 
                                    "parts": [{"text": text_to_add.strip()}]
                                })
                                current_user_text = ""
                            
                            # Accumulate model content
                            current_model_text += (" " if current_model_text else "") + content
                            last_role = "model"
                    
                    # Handle any remaining content
                    if current_user_text:
                        text_to_add = current_user_text
                        if system_message and len(contents) == 0:
                            text_to_add = system_message + text_to_add
                        contents.append({
                            "role": "user",
                            "parts": [{"text": text_to_add.strip()}]
                        })
                    elif current_model_text:
                        contents.append({
                            "role": "model",
                            "parts": [{"text": current_model_text.strip()}]
                        })
                    
                    # If no contents, create a simple one
                    if not contents:
                        contents = [{"role": "user", "parts": [{"text": (system_message + "Hello").strip()}]}]
                    
                    # Ensure conversation ends with user message (Gemini requirement)
                    if contents and contents[-1].get("role") == "model":
                        logger.warning("Conversation ends with model message, this may cause issues with Gemini API")
                    
                    # Debug logging
                    logger.info(f"Sending {len(contents)} messages to Gemini API")
                    for i, content in enumerate(contents):
                        role = content.get('role', 'unknown')
                        text_length = len(content['parts'][0]['text']) if content.get('parts') else 0
                        logger.debug(f"Message {i}: role={role}, text_length={text_length}")
                    
                    # Final validation - remove any empty messages
                    contents = [c for c in contents if c.get('parts') and c['parts'][0].get('text', '').strip()]
                    
                    # If still no valid contents after cleanup, create a simple fallback
                    if not contents:
                        contents = [{"role": "user", "parts": [{"text": "Hello"}]}]
                    
                    # Prepare request payload
                    payload = {
                        "contents": contents,
                        "generationConfig": {
                            "temperature": temperature,
                            "topP": top_p,
                            "maxOutputTokens": max_tokens
                        }
                    }
                    
                    # Log the payload for debugging
                    logger.debug(f"Gemini API payload: {json.dumps(payload, indent=2)}")
                    
                    headers = {
                        "x-goog-api-key": api_key,
                        "Content-Type": "application/json"
                    }
                    
                    # Try with a more stable model name if the configured one fails
                    model_to_use = self.model
                    if self.model == "gemini-2.5-flash":
                        model_to_use = "gemini-1.5-pro"  # More stable model
                    
                    # Make API call
                    response = requests.post(
                        f"{self.base_url}/{model_to_use}:generateContent",
                        headers=headers,
                        json=payload,
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Extract text from response
                        if "candidates" in result and result["candidates"]:
                            candidate = result["candidates"][0]
                            if "content" in candidate and "parts" in candidate["content"]:
                                parts = candidate["content"]["parts"]
                                if parts and "text" in parts[0]:
                                    return parts[0]["text"]
                        
                        # Fallback if structure is unexpected
                        logger.warning(f"Unexpected Gemini response structure: {result}")
                        return "I apologize, but I encountered an issue generating a response."
                    else:
                        # Log the actual error response for debugging
                        try:
                            error_detail = response.json()
                            logger.error(f"Gemini API error response: {error_detail}")
                        except:
                            logger.error(f"Gemini API error response (raw): {response.text}")
                        response.raise_for_status()
                
                start_time = time.time()
                response_text = await self._run_sync(_generate)
                end_time = time.time()
                
                logger.info(f"Gemini response generated in {end_time - start_time:.2f}s")
                return response_text
                
            except Exception as e:
                last_error = e
                self._handle_api_error(e, api_key)
                
                logger.warning(f"Gemini API attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < max_retries - 1:
                    # Wait before retrying
                    wait_time = (2 ** attempt) * 1  # Exponential backoff
                    logger.info(f"Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
        
        # If all retries failed
        raise Exception(f"Gemini API failed after {max_retries} attempts. Last error: {str(last_error)}")
    
    async def generate_response_streaming(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: int = 1000,
        temperature: float = 0.7,
        top_p: float = 0.9
    ):
        """
        Generate a streaming response using Gemini API.
        Note: This is a placeholder implementation as streaming requires different handling
        """
        # For now, we'll use the regular generate_response and yield it in chunks
        response = await self.generate_response(messages, max_tokens, temperature, top_p)
        
        # Simulate streaming by yielding chunks
        chunk_size = 50
        for i in range(0, len(response), chunk_size):
            yield response[i:i + chunk_size]
            await asyncio.sleep(0.05)  # Small delay to simulate streaming
    
    async def health_check(self) -> bool:
        """
        Check if the Gemini service is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Test with a simple request
            test_messages = [{"role": "user", "content": "Hello"}]
            response = await self.generate_response(test_messages)
            return len(response) > 0
            
        except Exception as e:
            logger.error(f"Gemini health check failed: {str(e)}")
            return False
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get information about the Gemini service."""
        return {
            "service_name": "Google Gemini API",
            "model": self.model,
            "api_keys_configured": len(self.api_keys),
            "api_keys_failed": len(self._failed_keys),
            "base_url": self.base_url
        }
    
    async def get_subscription_info(self) -> Dict[str, Any]:
        """Get Gemini API subscription information."""
        try:
            api_key = self._get_next_client_key()
            if not api_key:
                return {"error": "No API key available"}
            
            # Note: Gemini doesn't have a direct subscription endpoint
            # Return basic info
            return {
                "service": "Google Gemini API",
                "model": self.model,
                "status": "active",
                "rate_limits": "Standard limits apply"
            }
            
        except Exception as e:
            logger.error(f"Failed to get Gemini subscription info: {e}")
            return {"error": str(e)}
    
    def estimate_cost(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Estimate cost for Gemini API request."""
        # Calculate approximate token count
        total_chars = sum(len(msg["content"]) for msg in messages)
        estimated_tokens = total_chars // 4  # Rough estimate: 4 chars per token
        
        # Gemini pricing (approximate, as of 2024)
        # Flash models: Free tier available, then $0.075 per 1M input tokens
        input_cost = (estimated_tokens / 1_000_000) * 0.075
        
        return {
            "estimated_input_tokens": estimated_tokens,
            "estimated_cost_usd": round(input_cost, 6),
            "currency": "USD",
            "model": self.model,
            "pricing_note": "Gemini offers generous free tier limits"
        } 