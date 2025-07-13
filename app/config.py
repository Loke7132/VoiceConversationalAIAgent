from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    # Supabase Configuration
    supabase_url: str
    supabase_anon_key: str
    
    # Google Gemini AI Configuration - Multiple API keys for rotation
    gemini_api_keys: str  # Comma-separated API keys
    gemini_model: str = "gemini-2.5-flash"
    
    # Google Cloud Configuration
    google_cloud_project_id: str
    google_cloud_service_account_path: str
    google_cloud_default_voice: str = "en-US-Standard-A"
    google_cloud_language_code: str = "en-US"
    
    # Embedding Model Configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # RAG Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_results: int = 15  # Increased from 5 for better numerical comparisons
    
    # API Configuration
    api_title: str = "Voice Conversational Agentic AI"
    api_version: str = "1.0.0"
    debug: bool = False
    
    # File Upload Configuration
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_file_types: str = "pdf,txt,csv,json"  # Comma-separated file types
    
    # Conversation Configuration
    max_conversation_history: int = 20
    
    class Config:
        env_file = ".env"
        extra = "ignore"

    def get_gemini_api_keys(self) -> List[str]:
        """Parse comma-separated Gemini API keys."""
        return [key.strip() for key in self.gemini_api_keys.split(',') if key.strip()]
    
    def get_allowed_file_types(self) -> List[str]:
        """Parse comma-separated allowed file types."""
        return [ftype.strip().lower() for ftype in self.allowed_file_types.split(',') if ftype.strip()]


# Global settings instance
settings = Settings() 