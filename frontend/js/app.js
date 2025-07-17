// Voice Conversational AI Frontend JavaScript

class VoiceConversationalAI {
    constructor() {
        this.apiBaseUrl = 'http://localhost:8000';
        this.sessionId = this.generateSessionId();
        this.mapboxToken = 'pk.eyJ1Ijoic2FuamF5MDciLCJhIjoiY203M3M3ZHE3MDJ1cDJrcHh1aTZrd292NSJ9.J_dk3BJuJAW-wxqonaL8Xg'; // <-- Replace with real token
        this.map = null;
        this.propertyMarkers = {}; // stores markers by unique_id for quick access
        this.propertiesLoaded = false;
        this.isRecording = false;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.uploadedDocuments = 0;
        this.propertyList = [];
        
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
        this.setupAppointmentScheduling();
        // Preload property list for linkification
        this.preloadProperties();
        // Delegated click inside chat messages for "View Live Map" links
        const chatMsgContainer = document.getElementById('chatMessages');
        if (chatMsgContainer) {
            chatMsgContainer.addEventListener('click', (e) => {
                const link = e.target.closest('.view-property-link');
                if (link) {
                    e.preventDefault();
                    const propId = link.dataset.propId;
                    if (propId) {
                        this.showPropertyOnMap(propId);
                    }
                }
            });
        }
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

        // Floating Properties widget
        const propWidget = document.getElementById('propertiesWidget');
        if (propWidget) {
            propWidget.addEventListener('click', () => this.togglePropertiesMap());
        }
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
            
            // If the assistant indicates an appointment was scheduled, refresh the list
            if (/scheduled your appointment/i.test(data.llm_response)) {
                this.loadSessionAppointments();
            }
            
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
        
        // After building the formatted content, linkify property addresses
        const htmlOutput = result.join('');
        return this.linkifyProperties(htmlOutput);
        
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

    // Appointment Scheduling Methods
    setupAppointmentScheduling() {
        // View Associates button
        document.getElementById('viewAssociatesBtn').addEventListener('click', () => this.viewAssociates());
        
        // Schedule Appointment button
        document.getElementById('scheduleAppointmentBtn').addEventListener('click', () => this.showAppointmentForm());
        
        // Cancel appointment form
        document.getElementById('cancelAppointmentForm').addEventListener('click', () => this.hideAppointmentForm());
        
        // Appointment booking form
        document.getElementById('appointmentBookingForm').addEventListener('submit', (e) => this.handleAppointmentBooking(e));
        
        // Load associates on page load
        this.loadAssociates();
        
        // Load session appointments
        this.loadSessionAppointments();
    }

    async loadAssociates() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/associates`);
            
            if (response.ok) {
                const associates = await response.json();
                this.populateAssociateSelect(associates);
            } else {
                console.error('Failed to load associates');
            }
        } catch (error) {
            console.error('Error loading associates:', error);
        }
    }

    populateAssociateSelect(associates) {
        const associateSelect = document.getElementById('associateSelect');
        associateSelect.innerHTML = '<option value="">Select an associate...</option>';
        
        associates.forEach(associate => {
            const option = document.createElement('option');
            option.value = associate.id;
            option.textContent = `${associate.name} - ${associate.specialization}`;
            associateSelect.appendChild(option);
        });
    }

    async viewAssociates() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/associates`);
            
            if (response.ok) {
                const associates = await response.json();
                this.displayAssociates(associates);
            } else {
                console.error('Failed to load associates');
                this.showNotification('Failed to load associates', 'error');
            }
        } catch (error) {
            console.error('Error loading associates:', error);
            this.showNotification('Error loading associates', 'error');
        }
    }

    displayAssociates(associates) {
        const associatesList = document.getElementById('appointmentsList');
        associatesList.innerHTML = '';
        
        if (associates.length === 0) {
            associatesList.innerHTML = '<p class="no-appointments">No associates available</p>';
            return;
        }
        
        associates.forEach(associate => {
            const associateItem = document.createElement('div');
            associateItem.className = 'appointment-item';
            associateItem.innerHTML = `
                <div class="appointment-header">
                    <div class="appointment-title">${associate.name}</div>
                    <div class="appointment-status scheduled">${associate.specialization}</div>
                </div>
                <div class="appointment-details">
                    <div><i class="fas fa-envelope"></i> ${associate.email}</div>
                    <div><i class="fas fa-phone"></i> ${associate.phone || 'N/A'}</div>
                    <div><i class="fas fa-clock"></i> ${associate.availability_hours}</div>
                </div>
                <div class="appointment-actions">
                    <button class="btn btn-primary" onclick="app.showAvailability('${associate.id}')">
                        <i class="fas fa-calendar"></i> View Availability
                    </button>
                </div>
            `;
            associatesList.appendChild(associateItem);
        });
    }

    showAppointmentForm() {
        document.getElementById('appointmentForm').style.display = 'block';
        document.getElementById('availabilitySlots').style.display = 'none';
    }

    hideAppointmentForm() {
        document.getElementById('appointmentForm').style.display = 'none';
        document.getElementById('availabilitySlots').style.display = 'none';
    }

    async showAvailability(associateId) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/associates/${associateId}/availability?days_ahead=7`);
            
            if (response.ok) {
                const slots = await response.json();
                this.displayAvailabilitySlots(slots, associateId);
            } else {
                console.error('Failed to load availability');
                this.showNotification('Failed to load availability', 'error');
            }
        } catch (error) {
            console.error('Error loading availability:', error);
            this.showNotification('Error loading availability', 'error');
        }
    }

    displayAvailabilitySlots(slots, associateId) {
        const availabilityList = document.getElementById('availabilityList');
        const availabilitySlots = document.getElementById('availabilitySlots');
        
        availabilityList.innerHTML = '';
        availabilitySlots.style.display = 'block';
        
        if (slots.length === 0) {
            availabilityList.innerHTML = '<p class="no-appointments">No available slots</p>';
            return;
        }
        
        slots.forEach(slot => {
            const slotElement = document.createElement('div');
            slotElement.className = 'availability-slot';
            slotElement.dataset.associateId = associateId;
            slotElement.dataset.datetime = slot.datetime;
            
            const slotDate = new Date(slot.datetime);
            const formattedDate = slotDate.toLocaleDateString('en-US', {
                weekday: 'long',
                year: 'numeric',
                month: 'long',
                day: 'numeric'
            });
            const formattedTime = slotDate.toLocaleTimeString('en-US', {
                hour: '2-digit',
                minute: '2-digit'
            });
            
            slotElement.innerHTML = `
                <div class="availability-slot-time">${formattedDate} at ${formattedTime}</div>
                <div class="availability-slot-associate">${slot.duration_minutes} minutes</div>
            `;
            
            slotElement.addEventListener('click', () => this.selectTimeSlot(slotElement));
            availabilityList.appendChild(slotElement);
        });
    }

    selectTimeSlot(slotElement) {
        // Remove previous selection
        document.querySelectorAll('.availability-slot').forEach(slot => {
            slot.classList.remove('selected');
        });
        
        // Select new slot
        slotElement.classList.add('selected');
        
        // Fill appointment form
        const associateId = slotElement.dataset.associateId;
        const datetime = slotElement.dataset.datetime;
        
        document.getElementById('associateSelect').value = associateId;
        
        // Format datetime for input
        const date = new Date(datetime);
        const formattedDateTime = date.toISOString().slice(0, 16);
        document.getElementById('appointmentDate').value = formattedDateTime;
        
        // Show appointment form
        this.showAppointmentForm();
    }

    async handleAppointmentBooking(e) {
        e.preventDefault();
        
        const formData = new FormData(e.target);
        const appointmentData = {
            session_id: this.sessionId,
            associate_id: document.getElementById('associateSelect').value,
            user_name: document.getElementById('clientName').value,
            user_email: document.getElementById('clientEmail').value,
            user_phone: document.getElementById('clientPhone').value,
            scheduled_time: document.getElementById('appointmentDate').value,
            appointment_type: document.getElementById('appointmentType').value,
            notes: document.getElementById('appointmentNotes').value
        };
        
        try {
            const response = await fetch(`${this.apiBaseUrl}/schedule_appointment`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(appointmentData)
            });
            
            const result = await response.json();
            
            if (response.ok && result.success) {
                this.showNotification('Appointment scheduled successfully!', 'success');
                this.hideAppointmentForm();
                this.loadSessionAppointments();
                
                // Reset form
                document.getElementById('appointmentBookingForm').reset();
            } else {
                this.showNotification(result.message || 'Failed to schedule appointment', 'error');
            }
        } catch (error) {
            console.error('Error booking appointment:', error);
            this.showNotification('Error booking appointment', 'error');
        }
    }

    async loadSessionAppointments() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/appointments/session/${this.sessionId}`);
            
            if (response.ok) {
                const result = await response.json();
                this.displaySessionAppointments(result.appointments);
            } else {
                console.error('Failed to load session appointments');
            }
        } catch (error) {
            console.error('Error loading session appointments:', error);
        }
    }

    displaySessionAppointments(appointments) {
        const appointmentsList = document.getElementById('appointmentsList');
        appointmentsList.innerHTML = '';
        
        if (appointments.length === 0) {
            appointmentsList.innerHTML = '<p class="no-appointments">No upcoming appointments</p>';
            return;
        }
        
        appointments.forEach(appointment => {
            const appointmentItem = document.createElement('div');
            appointmentItem.className = 'appointment-item';
            
            const appointmentDate = new Date(appointment.scheduled_time);
            const formattedDate = appointmentDate.toLocaleDateString('en-US', {
                weekday: 'short',
                year: 'numeric',
                month: 'short',
                day: 'numeric'
            });
            const formattedTime = appointmentDate.toLocaleTimeString('en-US', {
                hour: '2-digit',
                minute: '2-digit'
            });
            
            appointmentItem.innerHTML = `
                <div class="appointment-header">
                    <div class="appointment-title">${appointment.appointment_type}</div>
                    <div class="appointment-status ${appointment.status}">${appointment.status}</div>
                </div>
                <div class="appointment-details">
                    <div><i class="fas fa-user"></i> ${appointment.associate_name}</div>
                    <div><i class="fas fa-calendar"></i> ${formattedDate} at ${formattedTime}</div>
                    <div><i class="fas fa-envelope"></i> ${appointment.user_email}</div>
                    ${appointment.notes ? `<div><i class="fas fa-sticky-note"></i> ${appointment.notes}</div>` : ''}
                </div>
                <div class="appointment-actions">
                    <button class="btn btn-secondary" onclick="app.cancelAppointment('${appointment.id}')">
                        <i class="fas fa-times"></i> Cancel
                    </button>
                </div>
            `;
            appointmentsList.appendChild(appointmentItem);
        });
    }

    async cancelAppointment(appointmentId) {
        if (!confirm('Are you sure you want to cancel this appointment?')) {
            return;
        }
        
        try {
            const response = await fetch(`${this.apiBaseUrl}/appointments/${appointmentId}/cancel`, {
                method: 'PUT'
            });
            
            const result = await response.json();
            
            if (response.ok && result.success) {
                this.showNotification('Appointment cancelled successfully', 'success');
                this.loadSessionAppointments();
            } else {
                this.showNotification(result.message || 'Failed to cancel appointment', 'error');
            }
        } catch (error) {
            console.error('Error cancelling appointment:', error);
            this.showNotification('Error cancelling appointment', 'error');
        }
    }

    showNotification(message, type = 'success') {
        const notification = document.createElement('div');
        notification.className = `appointment-notification ${type}`;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 5000);
    }

    /* ---------------- Property Map ---------------- */

    async togglePropertiesMap() {
        const overlay = document.getElementById('propertyMapOverlay');
        const mapContainer = document.getElementById('propertiesMap');
        const isVisible = overlay.style.display !== 'none';
        if (isVisible) {
            overlay.style.display = 'none';
            return;
        }

        overlay.style.display = 'flex';
        // No need to scroll; overlay is full screen

        // Lazy-init map
        if (!this.map) {
            if (!this.mapboxToken || this.mapboxToken.startsWith('YOUR_')) {
                this.showNotification('Mapbox token not configured in app.js', 'error');
                return;
            }
            mapboxgl.accessToken = this.mapboxToken;
            this.map = new mapboxgl.Map({
                container: 'propertiesMap',
                style: 'mapbox://styles/sanjay07/cm7apwi2u005d01p7efst9xll', // updated custom style
                center: [-73.9851, 40.7589], // Midtown Manhattan default
                zoom: 12
            });
            // Add navigation controls
            this.map.addControl(new mapboxgl.NavigationControl());
        }

        if (!this.propertiesLoaded) {
            await this.loadAndPlotProperties();
            this.propertiesLoaded = true;
        }
    }

    async loadAndPlotProperties() {
        try {
            const resp = await fetch(`${this.apiBaseUrl}/properties`);
            if (!resp.ok) {
                this.showNotification('Failed to load properties', 'error');
                return;
            }
            const props = await resp.json();
            this.propertyList = props; // store for linkification
            for (const prop of props) {
                await this.geocodeAndAddMarker(prop);
            }
        } catch (err) {
            console.error(err);
            this.showNotification('Error loading properties', 'error');
        }
    }

    async geocodeAndAddMarker(prop) {
        const query = encodeURIComponent(prop.address + ', New York, NY');
        const url = `https://api.mapbox.com/geocoding/v5/mapbox.places/${query}.json?access_token=${this.mapboxToken}&limit=1`;
        try {
            const res = await fetch(url);
            const data = await res.json();
            if (data && data.features && data.features.length > 0) {
                const [lng, lat] = data.features[0].center;
                const popup = new mapboxgl.Popup({ offset: 25 }).setText(`${prop.address}\n${prop.size_sf || ''} SF`);
                const marker = new mapboxgl.Marker()
                    .setLngLat([lng, lat])
                    .setPopup(popup)
                    .addTo(this.map);
                // Store reference for quick lookup when user clicks "View Live Map"
                if (prop.unique_id !== undefined) {
                    this.propertyMarkers[prop.unique_id] = { marker, coords: [lng, lat] };
                }
            }
        } catch (err) {
            console.error('Geocode error', err);
        }
    }

    /* Focus a particular property on the map given its unique_id */
    async showPropertyOnMap(propId) {
        // Ensure the map overlay is visible and markers are loaded
        await this.togglePropertiesMap();
        const entry = this.propertyMarkers[propId];
        if (!entry) {
            this.showNotification('Property not found on map', 'error');
            return;
        }
        const { coords, marker } = entry;
        // Smoothly move the map to the property location
        this.map.flyTo({ center: coords, zoom: 15, essential: true });
        if (marker && marker.togglePopup) {
            marker.togglePopup();
        }
    }

    /* Replace property addresses in assistant messages with live map links */
    linkifyProperties(htmlString) {
        if (!this.propertyList || this.propertyList.length === 0) return htmlString;
        let result = htmlString;
        const seen = new Set();
        // Sort longer addresses first to prevent partial replacements
        const sortedProps = [...this.propertyList].sort((a, b) => (b.address.length - a.address.length));
        for (const prop of sortedProps) {
            if (!prop.address) continue;
            const addrKey = prop.address.toLowerCase();
            if (seen.has(addrKey)) continue; // already added link for this address
            // Escape address for use in regex
            const escaped = prop.address.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            const regex = new RegExp(escaped, 'gi');
            if (regex.test(result)) {
                result = result.replace(regex, (match) => {
                    return `${match} <a href="#" class="view-property-link" data-prop-id="${prop.unique_id}">[View Map]</a>`;
                });
                seen.add(addrKey);
            }
        }
        return result;
    }

    /* Preload list of properties so we can linkify addresses even before map opens */
    async preloadProperties() {
        try {
            const resp = await fetch(`${this.apiBaseUrl}/properties`);
            if (resp.ok) {
                this.propertyList = await resp.json();
            }
        } catch (err) {
            console.warn('Could not preload properties:', err);
        }
    }
}

// Initialize the application
const app = new VoiceConversationalAI();

// Global functions for message actions
window.app = app; 