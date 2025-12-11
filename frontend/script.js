// API Configuration
// Note: For production, set API_BASE via environment variable or config file
// Current detection is for development convenience only
const API_BASE = window.location.hostname === 'localhost' 
    ? 'http://localhost:8000'
    : '/api';

// State
let currentChatId = null;
let chats = [];
let documents = [];

// DOM Elements
const chatList = document.getElementById('chatList');
const messagesContainer = document.getElementById('messagesContainer');
const queryInput = document.getElementById('queryInput');
const sendBtn = document.getElementById('sendBtn');
const newChatBtn = document.getElementById('newChatBtn');
const uploadBtn = document.getElementById('uploadBtn');
const fileInput = document.getElementById('fileInput');
const chatTitle = document.getElementById('chatTitle');
const deleteChatBtn = document.getElementById('deleteChatBtn');
const documentsList = document.getElementById('documentsList');
const refreshDocsBtn = document.getElementById('refreshDocsBtn');
const loadingOverlay = document.getElementById('loadingOverlay');
const loadingText = document.getElementById('loadingText');
const toast = document.getElementById('toast');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadChats();
    loadDocuments();
    setupEventListeners();
});

// Event Listeners
function setupEventListeners() {
    newChatBtn.addEventListener('click', createNewChat);
    sendBtn.addEventListener('click', sendQuery);
    uploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileUpload);
    deleteChatBtn.addEventListener('click', deleteCurrentChat);
    refreshDocsBtn.addEventListener('click', loadDocuments);
    
    queryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendQuery();
        }
    });
}

// API Functions
async function apiCall(endpoint, method = 'GET', data = null) {
    const options = {
        method,
        headers: {
            'Content-Type': 'application/json',
        },
    };
    
    if (data) {
        options.body = JSON.stringify(data);
    }
    
    const response = await fetch(`${API_BASE}${endpoint}`, options);
    
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Request failed');
    }
    
    return response.json();
}

// Chat Functions
async function loadChats() {
    try {
        chats = await apiCall('/chats');
        renderChats();
    } catch (error) {
        showToast('Failed to load chats', 'error');
        console.error(error);
    }
}

function renderChats() {
    chatList.innerHTML = '';
    
    if (chats.length === 0) {
        chatList.innerHTML = '<div style="padding: 20px; text-align: center; color: #95a5a6;">No chats yet</div>';
        return;
    }
    
    chats.forEach(chat => {
        const chatItem = document.createElement('div');
        chatItem.className = 'chat-item';
        if (chat.id === currentChatId) {
            chatItem.classList.add('active');
        }
        
        const date = new Date(chat.updated_at);
        const dateStr = date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
        
        chatItem.innerHTML = `
            <div class="chat-item-title">${chat.title}</div>
            <div class="chat-item-date">${dateStr}</div>
        `;
        
        chatItem.addEventListener('click', () => selectChat(chat.id));
        chatList.appendChild(chatItem);
    });
}

async function createNewChat() {
    try {
        const title = prompt('Enter chat title:', 'New Chat');
        if (!title) return;
        
        showLoading('Creating chat...');
        const chat = await apiCall('/chats', 'POST', { title });
        chats.unshift(chat);
        renderChats();
        selectChat(chat.id);
        hideLoading();
        showToast('Chat created successfully', 'success');
    } catch (error) {
        hideLoading();
        showToast('Failed to create chat', 'error');
        console.error(error);
    }
}

async function selectChat(chatId) {
    currentChatId = chatId;
    const chat = chats.find(c => c.id === chatId);
    
    if (chat) {
        chatTitle.textContent = chat.title;
        deleteChatBtn.style.display = 'block';
    }
    
    renderChats();
    await loadMessages(chatId);
    
    queryInput.disabled = false;
    sendBtn.disabled = false;
}

async function loadMessages(chatId) {
    try {
        showLoading('Loading messages...');
        const messages = await apiCall(`/chats/${chatId}/messages`);
        renderMessages(messages);
        hideLoading();
    } catch (error) {
        hideLoading();
        showToast('Failed to load messages', 'error');
        console.error(error);
    }
}

function renderMessages(messages) {
    messagesContainer.innerHTML = '';

    if (messages.length === 0) {
        messagesContainer.innerHTML = `
            <div class="welcome-message">
                <h2>Start asking questions!</h2>
                <p>Ask questions about your uploaded documents.</p>
            </div>
        `;
        return;
    }

    messages.forEach(message => {
        const elements = createMessageElement(message.role, message.content, {
            timestamp: message.created_at
        });
        messagesContainer.appendChild(elements.messageDiv);
    });

    scrollToBottom();
}

async function sendQuery() {
    if (!currentChatId || !queryInput.value.trim()) return;
    
    const query = queryInput.value.trim();
    queryInput.value = '';
    queryInput.disabled = true;
    sendBtn.disabled = true;
    
    // Add user message to UI immediately
    addMessageToUI('user', query);

    try {
        const assistantElements = addMessageToUI('assistant', '');
        await streamQuery(query, assistantElements);
        queryInput.disabled = false;
        sendBtn.disabled = false;
        queryInput.focus();
    } catch (error) {
        showToast('Failed to get response', 'error');
        console.error(error);
        queryInput.disabled = false;
        sendBtn.disabled = false;
    }
}

function addMessageToUI(role, content, sources = null) {
    const welcomeMsg = messagesContainer.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.remove();
    }
    
    const elements = createMessageElement(role, content, { sources });
    messagesContainer.appendChild(elements.messageDiv);
    scrollToBottom();
    return elements;
}

async function deleteCurrentChat() {
    if (!currentChatId) return;
    
    if (!confirm('Are you sure you want to delete this chat?')) return;
    
    try {
        showLoading('Deleting chat...');
        await apiCall(`/chats/${currentChatId}`, 'DELETE');
        
        chats = chats.filter(c => c.id !== currentChatId);
        currentChatId = null;
        
        chatTitle.textContent = 'Select a chat or create a new one';
        deleteChatBtn.style.display = 'none';
        messagesContainer.innerHTML = `
            <div class="welcome-message">
                <h2>Welcome to RAG Chat System</h2>
                <p>Upload documents and ask questions about them!</p>
            </div>
        `;
        
        queryInput.disabled = true;
        sendBtn.disabled = true;
        
        renderChats();
        hideLoading();
        showToast('Chat deleted successfully', 'success');
    } catch (error) {
        hideLoading();
        showToast('Failed to delete chat', 'error');
        console.error(error);
    }
}

// Document Functions
async function loadDocuments() {
    try {
        documents = await apiCall('/documents');
        renderDocuments();
    } catch (error) {
        showToast('Failed to load documents', 'error');
        console.error(error);
    }
}

function renderDocuments() {
    documentsList.innerHTML = '';
    
    if (documents.length === 0) {
        documentsList.innerHTML = '<div style="padding: 20px; text-align: center; color: #7f8c8d;">No documents uploaded</div>';
        return;
    }
    
    documents.forEach(doc => {
        const docItem = document.createElement('div');
        docItem.className = 'document-item';
        
        const date = new Date(doc.uploaded_at).toLocaleDateString();
        const status = doc.processed ? 'processed' : 'processing';
        const statusText = doc.processed ? `✓ ${doc.num_chunks} chunks` : 'Processing...';
        
        docItem.innerHTML = `
            <div class="document-name">${escapeHtml(doc.filename)}</div>
            <div class="document-info">${date}</div>
            <div class="document-status ${status}">${statusText}</div>
        `;
        
        documentsList.appendChild(docItem);
    });
}

async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        showLoading('Uploading and processing document...');
        
        const response = await fetch(`${API_BASE}/documents`, {
            method: 'POST',
            body: formData,
        });
        
        if (!response.ok) {
            throw new Error('Upload failed');
        }
        
        const document = await response.json();
        documents.unshift(document);
        renderDocuments();
        
        hideLoading();
        showToast('Document uploaded successfully', 'success');
    } catch (error) {
        hideLoading();
        showToast('Failed to upload document', 'error');
        console.error(error);
    }
    
    fileInput.value = '';
}

// Utility Functions
function showLoading(text = 'Loading...') {
    loadingText.textContent = text;
    loadingOverlay.style.display = 'flex';
}

function hideLoading() {
    loadingOverlay.style.display = 'none';
}

function showToast(message, type = 'info') {
    toast.textContent = message;
    toast.className = `toast ${type}`;
    toast.classList.add('show');
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function createMessageElement(role, content, options = {}) {
    const { sources = null, timestamp = null } = options;
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = renderMarkdown(content);

    const timeDiv = document.createElement('div');
    timeDiv.className = 'message-time';
    const date = timestamp ? new Date(timestamp) : new Date();
    timeDiv.textContent = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    messageDiv.appendChild(contentDiv);
    messageDiv.appendChild(timeDiv);

    let sourcesContainer = null;
    if (sources && sources.length > 0) {
        sourcesContainer = buildSourcesElement(sources);
        messageDiv.appendChild(sourcesContainer);
    }

    return { messageDiv, contentDiv, sourcesContainer, timeDiv };
}

function buildSourcesElement(sources) {
    const container = document.createElement('div');
    container.className = 'message-sources';

    const title = document.createElement('strong');
    title.textContent = 'Sources:';
    container.appendChild(title);

    sources.forEach((source, index) => {
        const isObject = source && typeof source === 'object';
        const label = isObject ? (source.label || `Source ${index + 1}`) : String(source);
        const contentText = isObject ? (source.content || '') : '';

        const details = document.createElement('details');
        const summary = document.createElement('summary');
        summary.textContent = label;
        details.appendChild(summary);

        const content = document.createElement('pre');
        content.textContent = contentText;
        if (contentText) {
            details.appendChild(content);
        }

        container.appendChild(details);
    });

    return container;
}

function renderMarkdown(text = '') {
    if (!text) {
        return '';
    }
    const html = marked.parse(text, { breaks: true });
    return DOMPurify.sanitize(html);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

async function streamQuery(query, assistantElements) {
    const response = await fetch(`${API_BASE}/query/stream`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            chat_id: currentChatId,
            query,
        }),
    });

    if (!response.ok || !response.body) {
        throw new Error('Streaming request failed');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let accumulated = '';

    while (true) {
        const { value, done } = await reader.read();
        if (done) {
            break;
        }

        buffer += decoder.decode(value, { stream: true });

        let separatorIndex;
        while ((separatorIndex = buffer.indexOf('\n\n')) !== -1) {
            const rawEvent = buffer.slice(0, separatorIndex).trim();
            buffer = buffer.slice(separatorIndex + 2);

            if (!rawEvent.startsWith('data:')) {
                continue;
            }

            const dataString = rawEvent.replace(/^data:\s*/, '');
            if (!dataString) {
                continue;
            }

            let payload;
            try {
                payload = JSON.parse(dataString);
            } catch (err) {
                console.error('Failed to parse SSE payload', err);
                continue;
            }

            if (payload.type === 'chunk') {
                if (payload.content) {
                    accumulated += payload.content;
                    assistantElements.contentDiv.innerHTML = renderMarkdown(accumulated);
                    scrollToBottom();
                }
            } else if (payload.type === 'end') {
                accumulated = payload.content || accumulated;
                assistantElements.contentDiv.innerHTML = renderMarkdown(accumulated);
                if (payload.sources && payload.sources.length > 0) {
                    if (assistantElements.sourcesContainer) {
                        assistantElements.sourcesContainer.remove();
                    }
                    assistantElements.sourcesContainer = buildSourcesElement(payload.sources);
                    assistantElements.messageDiv.appendChild(assistantElements.sourcesContainer);
                }
                scrollToBottom();
            } else if (payload.type === 'error') {
                throw new Error(payload.message || 'Streaming error');
            }
        }
    }
}
