// Voice Conversational AI Frontend JavaScript

class VoiceConversationalAI {
    constructor() {
        this.apiBaseUrl = 'http://localhost:8000';
        this.sessionId = this.generateSessionId();
        this.isRecording = false;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.uploadedDocuments = 0;
        
        this.init();
    }

    generateSessionId() {
        return 'user_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    init() {
        this.setupEventListeners();
        this.updateSessionDisplay();
        this.checkApiStatus();
        this.setupAudioRecording();
        this.loadAvailableVoices();
    }

    setupEventListeners() {
        // Chat functionality
        document.getElementById('sendBtn').addEventListener('click', () => this.sendMessage());
        document.getElementById('messageInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Voice functionality
        document.getElementById('voiceBtn').addEventListener('click', () => this.toggleRecording());
        document.getElementById('stopRecordingBtn').addEventListener('click', () => this.stopRecording());

        // File upload
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');

        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            this.handleFileUpload(e.dataTransfer.files);
        });

        fileInput.addEventListener('change', (e) => {
            this.handleFileUpload(e.target.files);
        });

        // Reset conversation
        document.getElementById('resetBtn').addEventListener('click', () => this.resetConversation());

        // Auto-resize textarea
        const messageInput = document.getElementById('messageInput');
        messageInput.addEventListener('input', () => {
            messageInput.style.height = 'auto';
            messageInput.style.height = Math.min(messageInput.scrollHeight, 100) + 'px';
        });
    }

    updateSessionDisplay() {
        document.getElementById('sessionId').textContent = this.sessionId;
    }

    async checkApiStatus() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/`);
            if (response.ok) {
                const data = await response.json();
                this.updateStatus('apiStatus', 'Online', 'text-success');
                this.updateStatus('chatStatus', 'Online', 'online');
                console.log('API Status:', data);
            } else {
                throw new Error('API not responding');
            }
        } catch (error) {
            this.updateStatus('apiStatus', 'Offline', 'text-error');
            this.updateStatus('chatStatus', 'Offline', 'offline');
            this.showToast('API connection failed', 'error');
        }
    }

    updateStatus(elementId, text, className) {
        const element = document.getElementById(elementId);
        element.textContent = text;
        element.className = element.className.split(' ')[0] + ' ' + className;
    }

    async setupAudioRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.audioStream = stream;
            console.log('Audio recording setup complete');
        } catch (error) {
            console.error('Error setting up audio recording:', error);
            this.showToast('Microphone access denied. Voice features disabled.', 'warning');
            document.getElementById('voiceBtn').disabled = true;
        }
    }

    async sendMessage() {
        const messageInput = document.getElementById('messageInput');
        const message = messageInput.value.trim();
        
        if (!message) return;

        // Clear input and add user message
        messageInput.value = '';
        messageInput.style.height = 'auto';
        this.addMessage(message, 'user');

        // Show loading
        this.showLoading('Generating response...');

        try {
            // Check if ReAct mode is enabled
            const reactMode = document.getElementById('reactMode').checked;
            const endpoint = reactMode ? '/converse_react' : '/chat';
            
            const response = await fetch(`${this.apiBaseUrl}${endpoint}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    message: message
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            // Add assistant response
            const responseText = reactMode ? data.llm_response : data.response;
            this.addMessage(responseText, 'assistant');
            
            // Show ReAct steps if available
            if (reactMode && data.metrics && data.metrics.react_steps) {
                this.addMessage(`🧠 ReAct processed this query with ${data.metrics.react_steps} reasoning steps`, 'system');
            }
            
            // Update metrics
            const processingTime = reactMode ? data.processing_time : data.processing_time;
            this.updateStatus('lastResponseTime', `${processingTime.toFixed(2)}s`, '');
            
            // Auto-play TTS if enabled
            if (document.getElementById('autoPlay').checked) {
                await this.playTTS(responseText);
            }

        } catch (error) {
            console.error('Error sending message:', error);
            this.addMessage('Sorry, I encountered an error. Please try again.', 'assistant', true);
            this.showToast('Failed to send message', 'error');
        } finally {
            this.hideLoading();
        }
    }

    async playTTS(text) {
        try {
            // Get voice ID from dropdown if it exists, otherwise use default
            const voiceSelectElement = document.getElementById('voiceSelect');
            const voiceId = voiceSelectElement ? voiceSelectElement.value : 'pNInz6obpgDQGcFmaJgB'; // Default voice ID
            
            const response = await fetch(`${this.apiBaseUrl}/speak`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: text,
                    voice_id: voiceId
                })
            });

            if (!response.ok) {
                throw new Error(`TTS error! status: ${response.status}`);
            }

            const data = await response.json();
            
            // Decode base64 audio and play
            const audioData = atob(data.audio_data);
            const audioArray = new Uint8Array(audioData.length);
            for (let i = 0; i < audioData.length; i++) {
                audioArray[i] = audioData.charCodeAt(i);
            }
            
            const audioBlob = new Blob([audioArray], { type: 'audio/wav' });
            const audioUrl = URL.createObjectURL(audioBlob);
            
            const audioPlayer = document.getElementById('audioPlayer');
            audioPlayer.src = audioUrl;
            audioPlayer.play();

        } catch (error) {
            console.error('Error playing TTS:', error);
            this.showToast('Failed to play audio', 'warning');
        }
    }

    toggleRecording() {
        if (this.isRecording) {
            this.stopRecording();
        } else {
            this.startRecording();
        }
    }

    async startRecording() {
        if (!this.audioStream) {
            this.showToast('Microphone not available', 'error');
            return;
        }

        try {
            this.audioChunks = [];
            this.mediaRecorder = new MediaRecorder(this.audioStream);
            
            this.mediaRecorder.ondataavailable = (event) => {
                this.audioChunks.push(event.data);
            };

            this.mediaRecorder.onstop = () => {
                this.processRecording();
            };

            this.mediaRecorder.start();
            this.isRecording = true;

            // Update UI
            document.getElementById('voiceBtn').classList.add('recording');
            document.getElementById('voiceControls').classList.remove('hidden');
            
            const isMultilingualMode = document.getElementById('multilingualMode').checked;
            const placeholder = isMultilingualMode ? 
                'Recording multilingual voice message...' : 
                'Recording voice message...';
            document.getElementById('messageInput').placeholder = placeholder;

        } catch (error) {
            console.error('Error starting recording:', error);
            this.showToast('Failed to start recording', 'error');
        }
    }

    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;

            // Update UI
            document.getElementById('voiceBtn').classList.remove('recording');
            document.getElementById('voiceControls').classList.add('hidden');
            document.getElementById('messageInput').placeholder = 'Type your message here...';
        }
    }

    async processRecording() {
        if (this.audioChunks.length === 0) return;

        // Check modes
        const isMultilingualMode = document.getElementById('multilingualMode').checked;
        const isReactMode = document.getElementById('reactMode').checked;
        
        // Determine loading text based on modes
        let loadingText = 'Processing voice message...';
        if (isReactMode && isMultilingualMode) {
            loadingText = 'Processing multilingual voice message with ReAct reasoning...';
        } else if (isReactMode) {
            loadingText = 'Processing voice message with ReAct reasoning...';
        } else if (isMultilingualMode) {
            loadingText = 'Processing multilingual voice message...';
        }
        
        this.showLoading(loadingText);

        try {
            // Create audio blob
            const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
            
            // Determine endpoint based on modes
            let endpoint;
            if (isReactMode && isMultilingualMode) {
                // TODO: Create multilingual ReAct endpoint if needed
                endpoint = '/converse_react_voice'; // For now, use ReAct voice (English only)
            } else if (isReactMode) {
                endpoint = '/converse_react_voice';
            } else if (isMultilingualMode) {
                endpoint = '/converse_multilingual';
            } else {
                endpoint = '/converse';
            }
            
            // Send to backend for end-to-end conversation
            const formData = new FormData();
            formData.append('audio', audioBlob, 'recording.wav');
            formData.append('session_id', this.sessionId);

            const response = await fetch(`${this.apiBaseUrl}${endpoint}`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Conversation error! status: ${response.status}`);
            }

            const data = await response.json();
            
            // Add transcribed message as user message
            this.addMessage(data.transcription, 'user');
            
            // Show language detection info if multilingual mode is enabled
            if (isMultilingualMode && data.metrics && data.metrics.detected_language) {
                this.showToast(`Language detected: ${data.metrics.detected_language}`, 'info');
            }
            
            // Show ReAct steps if ReAct mode is enabled
            if (isReactMode && data.metrics && data.metrics.react_steps) {
                this.addMessage(`🧠 ReAct processed this query with ${data.metrics.react_steps} reasoning steps`, 'system');
            }
            
            // Add AI response
            this.addMessage(data.llm_response, 'assistant');
            
            // Play audio response
            if (data.audio_response) {
                const audioData = atob(data.audio_response);
                const audioArray = new Uint8Array(audioData.length);
                for (let i = 0; i < audioData.length; i++) {
                    audioArray[i] = audioData.charCodeAt(i);
                }
                
                const audioBlob = new Blob([audioArray], { type: 'audio/wav' });
                const audioUrl = URL.createObjectURL(audioBlob);
                
                const audioPlayer = document.getElementById('audioPlayer');
                audioPlayer.src = audioUrl;
                audioPlayer.play();
            }
            
            // Update metrics
            this.updateStatus('lastResponseTime', `${data.processing_time.toFixed(2)}s`, '');

        } catch (error) {
            console.error('Error processing recording:', error);
            this.addMessage('Sorry, I couldn\'t process your voice message. Please try again.', 'assistant', true);
            this.showToast('Failed to process voice message', 'error');
        } finally {
            this.hideLoading();
        }
    }

    async handleFileUpload(files) {
        if (files.length === 0) return;

        this.showLoading('Uploading documents...');

        try {
            const formData = new FormData();
            
            // Add all files to form data
            for (let file of files) {
                formData.append('files', file);
            }

            const response = await fetch(`${this.apiBaseUrl}/upload_rag_docs`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Upload error! status: ${response.status}`);
            }

            const data = await response.json();
            
            // Process results
            let successCount = 0;
            let errorCount = 0;
            
            data.results.forEach(result => {
                this.addFileToUI(result);
                if (result.success) {
                    successCount++;
                } else {
                    errorCount++;
                }
            });

            this.uploadedDocuments += successCount;
            this.updateStatus('documentCount', this.uploadedDocuments.toString(), '');

            if (successCount > 0) {
                this.showToast(`Successfully uploaded ${successCount} document(s)`, 'success');
            }
            if (errorCount > 0) {
                this.showToast(`Failed to upload ${errorCount} document(s)`, 'error');
            }

        } catch (error) {
            console.error('Error uploading files:', error);
            this.showToast('Failed to upload documents', 'error');
        } finally {
            this.hideLoading();
        }
    }

    addFileToUI(result) {
        const uploadedFiles = document.getElementById('uploadedFiles');
        
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        
        const icon = this.getFileIcon(result.filename);
        const statusClass = result.success ? 'success' : 'error';
        const statusText = result.success ? 'Processed' : 'Error';
        
        fileItem.innerHTML = `
            <i class="fas ${icon}"></i>
            <div class="file-info">
                <div class="file-name">${result.filename}</div>
                <div class="file-status ${statusClass}">${statusText}</div>
            </div>
        `;
        
        uploadedFiles.appendChild(fileItem);
    }

    getFileIcon(filename) {
        const extension = filename.split('.').pop().toLowerCase();
        switch (extension) {
            case 'pdf': return 'fa-file-pdf';
            case 'txt': return 'fa-file-text';
            case 'csv': return 'fa-file-csv';
            case 'json': return 'fa-file-code';
            default: return 'fa-file';
        }
    }

    async resetConversation() {
        if (!confirm('Are you sure you want to reset the conversation?')) return;

        this.showLoading('Resetting conversation...');

        try {
            const formData = new FormData();
            formData.append('session_id', this.sessionId);

            const response = await fetch(`${this.apiBaseUrl}/reset`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Reset error! status: ${response.status}`);
            }

            // Clear chat messages
            const chatMessages = document.getElementById('chatMessages');
            chatMessages.innerHTML = `
                <div class="message assistant">
                    <div class="message-avatar">
                        <i class="fas fa-robot"></i>
                    </div>
                    <div class="message-content">
                        <p>Conversation reset! I'm ready to help you again. How can I assist you today?</p>
                        <div class="message-time">Just now</div>
                    </div>
                </div>
            `;

            this.showToast('Conversation reset successfully', 'success');

        } catch (error) {
            console.error('Error resetting conversation:', error);
            this.showToast('Failed to reset conversation', 'error');
        } finally {
            this.hideLoading();
        }
    }

    addMessage(content, sender, isError = false) {
        const chatMessages = document.getElementById('chatMessages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        
        const avatarIcon = sender === 'user' ? 'fa-user' : (sender === 'system' ? 'fa-brain' : 'fa-robot');
        const timestamp = new Date().toLocaleTimeString();
        
        // Enhanced content parsing for structured formatting
        let formattedContent = content;
        
        if (sender === 'assistant') {
            // Parse and format structured content
            formattedContent = this.parseStructuredContent(content);
        } else if (sender === 'system') {
            // For system messages, just escape HTML and add styling
            formattedContent = `<p>${this.escapeHtml(content)}</p>`;
        } else {
            // For user messages, just escape HTML
            formattedContent = `<p>${this.escapeHtml(content)}</p>`;
        }
        
        const messageActions = sender === 'assistant' ? `
            <div class="message-actions">
                <button class="message-action" onclick="app.playTTS('${content.replace(/'/g, "\\'")}')" title="Play audio">
                    <i class="fas fa-volume-up"></i>
                </button>
                <button class="message-action" onclick="app.copyToClipboard('${content.replace(/'/g, "\\'")}')" title="Copy text">
                    <i class="fas fa-copy"></i>
                </button>
            </div>
        ` : '';
        
        messageDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fas ${avatarIcon}"></i>
            </div>
            <div class="message-content ${isError ? 'error' : ''}">
                ${formattedContent}
                <div class="message-time">${timestamp}</div>
                ${messageActions}
            </div>
        `;
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    parseStructuredContent(content) {
        // Escape HTML first
        let formatted = this.escapeHtml(content);
        
        // Parse bold text (**text** or __text__)
        formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        formatted = formatted.replace(/__(.*?)__/g, '<strong>$1</strong>');
        
        // Split content into lines for processing
        const lines = formatted.split('\n');
        const result = [];
        let inList = false;
        let listType = null;
        let listItems = [];
        
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();
            
            if (line === '') {
                // Empty line - close any open list and add line break
                if (inList) {
                    result.push(this.formatList(listItems, listType));
                    inList = false;
                    listItems = [];
                }
                continue; // Skip empty lines instead of adding breaks
            }
            
            // Check for bullet points (• or -)
            if (line.match(/^[•\-]\s+(.+)/)) {
                const content = line.replace(/^[•\-]\s+/, '');
                if (!inList || listType !== 'ul') {
                    if (inList) {
                        result.push(this.formatList(listItems, listType));
                    }
                    inList = true;
                    listType = 'ul';
                    listItems = [];
                }
                listItems.push(content);
                continue;
            }
            
            // Check for numbered lists (1. 2. 3. etc.)
            if (line.match(/^\d+\.\s+(.+)/)) {
                const content = line.replace(/^\d+\.\s+/, '');
                if (!inList || listType !== 'ol') {
                    if (inList) {
                        result.push(this.formatList(listItems, listType));
                    }
                    inList = true;
                    listType = 'ol';
                    listItems = [];
                }
                listItems.push(content);
                continue;
            }
            
            // Check for nested items (indented with spaces or tabs)
            if (line.match(/^\s{2,}[•\-]\s+(.+)/) && inList) {
                const content = line.replace(/^\s+[•\-]\s+/, '');
                // Add as nested item to the last list item
                if (listItems.length > 0) {
                    listItems[listItems.length - 1] += `<ul><li>${content}</li></ul>`;
                }
                continue;
            }
            
            // Check for property header patterns (bold address with details)
            if (line.match(/^\d+\.\s+\*\*.*\*\*/)) {
                // This is a numbered property with bold header
                if (inList) {
                    result.push(this.formatList(listItems, listType));
                    inList = false;
                    listItems = [];
                }
                result.push(`<div class="property-details">${line}</div>`);
                continue;
            }
            
            // Regular text line
            if (inList) {
                result.push(this.formatList(listItems, listType));
                inList = false;
                listItems = [];
            }
            
            // Add as paragraph
            if (line.length > 0) {
                result.push(`<p>${line}</p>`);
            }
        }
        
        // Close any remaining list
        if (inList) {
            result.push(this.formatList(listItems, listType));
        }
        
        return result.join('');
    }
    
    formatList(items, type) {
        if (items.length === 0) return '';
        
        const tag = type === 'ol' ? 'ol' : 'ul';
        const listItems = items.map(item => `<li>${item}</li>`).join('');
        return `<${tag}>${listItems}</${tag}>`;
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    async copyToClipboard(text) {
        try {
            await navigator.clipboard.writeText(text);
            this.showToast('Text copied to clipboard', 'success');
        } catch (error) {
            console.error('Error copying to clipboard:', error);
            this.showToast('Failed to copy text', 'error');
        }
    }

    showLoading(text = 'Loading...') {
        document.getElementById('loadingText').textContent = text;
        document.getElementById('loadingOverlay').classList.remove('hidden');
    }

    hideLoading() {
        document.getElementById('loadingOverlay').classList.add('hidden');
    }

    showToast(message, type = 'info') {
        const toastContainer = document.getElementById('toastContainer');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        const icons = {
            success: 'fa-check-circle',
            error: 'fa-exclamation-circle',
            warning: 'fa-exclamation-triangle',
            info: 'fa-info-circle'
        };
        
        toast.innerHTML = `
            <i class="fas ${icons[type] || icons.info}"></i>
            <span>${message}</span>
        `;
        
        toastContainer.appendChild(toast);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 5000);
        
        // Add click to dismiss
        toast.addEventListener('click', () => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        });
    }

    async loadAvailableVoices() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/supported_languages`);
            
            if (response.ok) {
                const data = await response.json();
                const voiceSelect = document.getElementById('voiceSelect');
                
                if (voiceSelect && data.voice_availability) {
                    // Clear existing options
                    voiceSelect.innerHTML = '';
                    
                    // Add default option
                    const defaultOption = document.createElement('option');
                    defaultOption.value = 'en-US-Standard-A';
                    defaultOption.textContent = 'English (US) - Standard';
                    voiceSelect.appendChild(defaultOption);
                    
                    // Group voices by language
                    const languages = Object.keys(data.voice_availability).sort();
                    
                    languages.forEach(langCode => {
                        const voices = data.voice_availability[langCode];
                        const langName = data.supported_languages[langCode] || langCode.toUpperCase();
                        
                        // Create optgroup for language
                        const optgroup = document.createElement('optgroup');
                        optgroup.label = langName;
                        
                        voices.forEach(voice => {
                            const option = document.createElement('option');
                            option.value = voice.voice_name;
                            option.textContent = `${voice.voice_name} (${voice.gender})`;
                            optgroup.appendChild(option);
                        });
                        
                        voiceSelect.appendChild(optgroup);
                    });
                    
                    console.log(`Loaded ${languages.length} languages with voice support`);
                }
            } else {
                console.warn('Failed to load available voices, using default');
            }
        } catch (error) {
            console.error('Error loading available voices:', error);
            // Keep default voice if loading fails
        }
    }
}

// Initialize the application
const app = new VoiceConversationalAI();

// Global functions for message actions
window.app = app; 