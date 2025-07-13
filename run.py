#!/usr/bin/env python3
"""
Run script for Voice Conversational Agentic AI API
This script starts the FastAPI application with proper configuration.
"""

import uvicorn
import os
import sys
from pathlib import Path

# Add the project directory to Python path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

def main():
    """Main function to run the application."""
    
    # Check if .env file exists
    env_file = project_dir / ".env"
    if not env_file.exists():
        print("⚠️  .env file not found!")
        print("Please create a .env file based on env_template.txt")
        print("Make sure to add your API keys and Supabase configuration.")
        return
    
    print("🚀 Starting Voice Conversational Agentic AI API...")
    print("📖 API Documentation will be available at: http://localhost:8000/docs")
    print("📱 ReDoc Documentation will be available at: http://localhost:8000/redoc")
    print("🔧 Root endpoint: http://localhost:8000/")
    print()
    
    # Configure uvicorn
    config = {
        "app": "app.main:app",
        "host": "0.0.0.0",
        "port": 8000,
        "reload": True,
        "log_level": "info",
        "access_log": True,
    }
    
    # Check if we're in development mode
    if os.getenv("DEBUG", "false").lower() == "true":
        config["log_level"] = "debug"
        print("🐛 Debug mode enabled")
    
    try:
        uvicorn.run(**config)
    except KeyboardInterrupt:
        print("\n👋 Shutting down Voice Conversational Agentic AI API...")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 