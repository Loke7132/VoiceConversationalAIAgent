# Voice Conversational AI - Frontend

A beautiful, modern web interface for the Voice Conversational Agentic AI system. This frontend provides an intuitive chat interface with voice recording, file upload, and real-time communication with the AI backend.

## Features

### 🎯 **Core Functionality**
- **Text Chat**: Type messages and get AI responses with RAG context
- **Voice Conversation**: Record voice messages and get spoken responses
- **Document Upload**: Drag & drop documents for RAG knowledge base
- **Session Management**: Persistent conversation history
- **Real-time Status**: Live API status and performance metrics

### 🎨 **User Interface**
- **Modern Design**: Clean, professional interface with gradients and animations
- **Responsive Layout**: Works on desktop, tablet, and mobile devices
- **Dark/Light Theme**: Comfortable viewing experience
- **Intuitive Controls**: Easy-to-use buttons and interactions
- **Toast Notifications**: Real-time feedback for user actions

### 🔊 **Voice Features**
- **Voice Recording**: Click-to-record voice messages
- **Audio Playback**: Listen to AI responses with natural voices
- **Voice Selection**: Choose from multiple ElevenLabs voices
- **Auto-play**: Optional automatic audio playback
- **Real-time Processing**: Live transcription and TTS

### 📄 **Document Management**
- **File Upload**: Support for PDF, TXT, CSV, JSON files
- **Drag & Drop**: Intuitive file upload interface
- **Upload Progress**: Visual feedback during processing
- **File Status**: Track successful and failed uploads
- **Document Counter**: See how many documents are indexed

## Setup Instructions

### Prerequisites
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Voice Conversational AI backend running on `http://localhost:8000`
- Microphone access for voice features (optional)

### Installation

1. **Download the frontend files**
   ```bash
   # The frontend files should be in the `frontend/` directory
   frontend/
   ├── index.html
   ├── css/
   │   └── style.css
   ├── js/
   │   └── app.js
   └── README.md
   ```

2. **Start the backend API**
   ```bash
   # Make sure your backend is running
   python run.py
   # or
   uvicorn app.main:app --reload
   ```

3. **Open the frontend**
   - **Option 1**: Open `frontend/index.html` directly in your browser
   - **Option 2**: Serve via HTTP server (recommended for voice features)
     ```bash
     # Using Python
     cd frontend
     python -m http.server 8080
     # Then visit http://localhost:8080
     
     # Using Node.js
     npx serve . -p 8080
     
     # Using any other static file server
     ```

### Browser Permissions

For full functionality, allow the following permissions:
- **Microphone Access**: Required for voice recording features
- **Audio Playback**: Required for TTS responses

## Usage Guide

### Text Chat
1. Type your message in the text area at the bottom
2. Click the send button (✈️) or press Enter
3. Wait for the AI response
4. View conversation history in the chat area

### Voice Conversation
1. Click the microphone button (🎤) to start recording
2. Speak your message clearly
3. Click the stop button to end recording
4. The system will transcribe, process, and respond with voice

### Document Upload
1. Click on the upload area or drag files directly
2. Select PDF, TXT, CSV, or JSON files
3. Wait for processing to complete
4. Files will appear in the uploaded documents list
5. Ask questions about uploaded content

### Settings
- **Voice Selection**: Choose from different ElevenLabs voices
- **Auto-play**: Toggle automatic audio playback for responses
- **Session Reset**: Clear conversation history

## Features Overview

### Chat Interface
- **Message History**: Scrollable conversation view
- **Typing Indicators**: Visual feedback during AI processing
- **Message Actions**: Copy text, play audio for each message
- **Timestamps**: See when each message was sent
- **Error Handling**: Clear error messages and retry options

### Voice Controls
- **Recording Indicator**: Visual feedback during recording
- **Voice Activity**: Animated microphone during recording
- **Audio Quality**: High-quality voice processing
- **Multi-format Support**: Various audio formats supported

### File Management
- **Drag & Drop**: Intuitive file upload
- **Progress Tracking**: Visual upload progress
- **File Validation**: Automatic format checking
- **Status Updates**: Real-time processing feedback
- **Error Recovery**: Retry failed uploads

### Performance Monitoring
- **API Status**: Real-time backend connectivity
- **Response Times**: Track processing performance
- **Document Count**: Monitor RAG knowledge base size
- **Connection Quality**: Network status indicators

## Customization

### Styling
The frontend uses CSS variables for easy theming. Edit `css/style.css`:

```css
:root {
    --primary-color: #667eea;    /* Main theme color */
    --secondary-color: #764ba2;  /* Secondary theme color */
    --accent-color: #f093fb;     /* Accent color */
    /* ... other variables */
}
```

### API Configuration
Update the API base URL in `js/app.js`:

```javascript
constructor() {
    this.apiBaseUrl = 'http://localhost:8000'; // Change this URL
    // ...
}
```

### Voice Settings
Modify voice options in `index.html`:

```html
<select id="voiceSelect" class="form-control">
    <option value="21m00Tcm4TlvDq8ikWAM">Rachel (Default)</option>
    <option value="your-voice-id">Your Custom Voice</option>
</select>
```

## Troubleshooting

### Common Issues

1. **API Connection Failed**
   - Ensure backend is running on `http://localhost:8000`
   - Check CORS settings in backend
   - Verify network connectivity

2. **Microphone Not Working**
   - Grant microphone permissions in browser
   - Check microphone hardware
   - Try refreshing the page

3. **Files Not Uploading**
   - Check file format (PDF, TXT, CSV, JSON only)
   - Verify file size limits
   - Ensure backend is processing uploads

4. **Audio Not Playing**
   - Check browser audio permissions
   - Verify speakers/headphones
   - Check TTS service availability

### Browser Compatibility
- **Chrome**: Full support (recommended)
- **Firefox**: Full support
- **Safari**: Full support (may need HTTPS for microphone)
- **Edge**: Full support
- **Mobile Browsers**: Basic support (limited voice features)

### HTTPS Requirements
Some browsers require HTTPS for microphone access. For production:
1. Deploy with SSL certificate
2. Use `https://` URLs
3. Update API URLs accordingly

## Architecture

### Frontend Components
```
Frontend Structure:
├── HTML (index.html)
│   ├── Header (title, controls)
│   ├── Sidebar (upload, settings)
│   ├── Chat Area (messages, input)
│   └── Status Bar (metrics)
├── CSS (style.css)
│   ├── Layout & Grid
│   ├── Components
│   ├── Animations
│   └── Responsive Design
└── JavaScript (app.js)
    ├── VoiceConversationalAI Class
    ├── API Communication
    ├── Voice Recording
    ├── File Upload
    └── UI Management
```

### Data Flow
1. **User Input** → Frontend Interface
2. **Frontend** → API Request → Backend
3. **Backend** → AI Processing → API Response
4. **Frontend** → UI Update → User Experience

## Development

### Adding New Features
1. **HTML**: Add new elements with appropriate IDs
2. **CSS**: Style new components following existing patterns
3. **JavaScript**: Extend the `VoiceConversationalAI` class
4. **API**: Update backend integration as needed

### Code Structure
- **Modular Design**: Easy to extend and maintain
- **Error Handling**: Comprehensive error management
- **Performance**: Optimized for real-time interaction
- **Accessibility**: Keyboard navigation and screen reader support

## Demo Tips

### For Hackathon Presentation
1. **Preparation**:
   - Test all features beforehand
   - Prepare sample documents to upload
   - Have test questions ready

2. **Demonstration Flow**:
   - Show text chat with simple questions
   - Upload a document and ask related questions
   - Demonstrate voice conversation
   - Highlight real-time features

3. **Impressive Features**:
   - End-to-end voice conversation
   - Document upload and RAG integration
   - Real-time performance metrics
   - Modern, professional interface

## Support

For issues or questions:
1. Check the console for error messages
2. Verify backend API is running
3. Review network requests in browser dev tools
4. Check microphone and audio permissions

## License

This frontend is part of the Voice Conversational Agentic AI project and follows the same licensing terms. 