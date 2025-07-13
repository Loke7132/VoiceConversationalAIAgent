#!/usr/bin/env python3
"""
Simple HTTP server for Voice Conversational AI Frontend
This script serves the frontend files with proper CORS headers for development.
"""

import http.server
import socketserver
import webbrowser
import os
import sys
from pathlib import Path


class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler with CORS support."""
    
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()


def main():
    """Start the frontend server."""
    
    # Change to frontend directory
    frontend_dir = Path(__file__).parent
    os.chdir(frontend_dir)
    
    # Configuration
    port = 8080
    host = 'localhost'
    
    print("🌐 Voice Conversational AI - Frontend Server")
    print("=" * 50)
    print(f"📁 Serving from: {frontend_dir}")
    print(f"🌍 Server URL: http://{host}:{port}")
    print(f"🔧 Backend API: http://localhost:8000 (ensure it's running)")
    print()
    
    # Check if index.html exists
    if not (frontend_dir / 'index.html').exists():
        print("❌ Error: index.html not found!")
        print("   Make sure you're running this from the frontend directory.")
        sys.exit(1)
    
    try:
        # Create server
        with socketserver.TCPServer((host, port), CORSHTTPRequestHandler) as httpd:
            print(f"🚀 Starting server on http://{host}:{port}")
            print("   Press Ctrl+C to stop the server")
            print()
            
            # Open browser automatically
            try:
                webbrowser.open(f'http://{host}:{port}')
                print("🌐 Opening browser automatically...")
            except:
                print("💡 Manual: Open your browser and go to the URL above")
            
            print()
            print("📋 Frontend Features:")
            print("   ✅ Text chat with AI")
            print("   ✅ Voice conversation (requires microphone)")
            print("   ✅ Document upload for RAG")
            print("   ✅ Real-time status monitoring")
            print("   ✅ Responsive design")
            print()
            print("🎯 Ready for demo! Make sure your backend is running.")
            print()
            
            # Start serving
            httpd.serve_forever()
            
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"❌ Error: Port {port} is already in use!")
            print(f"   Try a different port or stop the process using port {port}")
        else:
            print(f"❌ Error starting server: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down frontend server...")
        print("   Thanks for using Voice Conversational AI!")


if __name__ == "__main__":
    main() 