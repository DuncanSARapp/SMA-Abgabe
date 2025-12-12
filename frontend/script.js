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
        
        // Auto-select first chat if none is active
        if (!currentChatId && chats.length > 0) {
            await selectChat(chats[0].id);
        }
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

async function ensureActiveChat() {
    // If a chat is already active, nothing to do
    if (currentChatId) return currentChatId;

    // If chats were loaded, pick the first one
    if (chats.length > 0) {
        await selectChat(chats[0].id);
        return currentChatId;
    }

    // Otherwise create a fallback chat so the user can start typing immediately
    try {
        showLoading('Creating chat...');
        const chat = await apiCall('/chats', 'POST', { title: `Chat ${chats.length + 1}` });
        chats.unshift(chat);
        renderChats();
        await selectChat(chat.id);
        showToast('New chat created', 'success');
        return chat.id;
    } finally {
        hideLoading();
    }
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
    if (!queryInput.value.trim()) return;

    // Make sure there's an active chat, create/select one if needed
    try {
        await ensureActiveChat();
    } catch (error) {
        showToast('Failed to prepare chat', 'error');
        console.error(error);
        return;
    }
    
    const query = queryInput.value.trim();
    queryInput.value = '';
    queryInput.disabled = true;
    sendBtn.disabled = true;
    
    // Add user message to UI immediately
    addMessageToUI('user', query);

    try {
        const assistantElements = addMessageToUI('assistant', '');
        await streamQuery(query, assistantElements);
    } catch (error) {
        const noDocsMessage = 'No active documents selected for querying';
        if (error?.message && error.message.toLowerCase().includes(noDocsMessage.toLowerCase())) {
            addAssistantInfoMessage('Bitte aktiviere mindestens ein Dokument in der rechten Liste (grüner Haken), bevor du eine Frage stellst.');
        } else {
            showToast('Failed to get response', 'error');
            console.error(error);
        }
    } finally {
        queryInput.disabled = false;
        sendBtn.disabled = false;
        queryInput.focus();
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

function addAssistantInfoMessage(content) {
    const elements = addMessageToUI('assistant', content);
    if (elements.thinkingContainer) {
        elements.thinkingContainer.remove();
        elements.thinkingContainer = null;
    }
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
        
        queryInput.disabled = false;
        sendBtn.disabled = false;
        
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
        const isProcessed = doc.processed;
        const status = isProcessed ? 'processed' : 'unprocessed';
        const statusText = isProcessed ? `✓ ${doc.num_chunks} chunks` : '⚠ Not in vector store';
        const queryPillClass = doc.query_enabled ? 'query-pill active' : 'query-pill inactive';
        const toggleTitle = doc.query_enabled ? 'Included in retrieval' : 'Excluded from retrieval';
        
        docItem.innerHTML = `
            <div class="document-name">${escapeHtml(doc.filename)}</div>
            <div class="document-info-row">
                <span>${date}</span>
                <span class="document-collection">${escapeHtml(doc.collection_name)}</span>
            </div>
            <div class="document-status-row">
                <div class="document-status ${status}">${statusText}</div>
                <div class="${queryPillClass}">${doc.query_enabled ? 'Query an' : 'Query aus'}</div>
            </div>
            <div class="document-actions">
                ${!isProcessed ? `<button class="btn-reprocess" data-id="${doc.id}">🔄 Reprocess</button>` : ''}
                <button class="doc-toggle ${doc.query_enabled ? 'active' : ''}" data-id="${doc.id}" aria-pressed="${doc.query_enabled}" title="${toggleTitle}">${doc.query_enabled ? '✔' : ''}</button>
                <button class="btn-delete-doc" data-id="${doc.id}">🗑️</button>
            </div>
        `;
        
        // Add event listeners
        const reprocessBtn = docItem.querySelector('.btn-reprocess');
        if (reprocessBtn) {
            reprocessBtn.addEventListener('click', () => reprocessDocument(doc.id));
        }
        const toggleBtn = docItem.querySelector('.doc-toggle');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', () => toggleDocumentQuery(doc.id, doc.query_enabled));
        }
        
        const deleteBtn = docItem.querySelector('.btn-delete-doc');
        if (deleteBtn) {
            deleteBtn.addEventListener('click', () => deleteDocument(doc.id));
        }
        
        documentsList.appendChild(docItem);
    });
}

async function reprocessDocument(docId) {
    try {
        showLoading('Reprocessing document...');
        const response = await fetch(`${API_BASE}/documents/${docId}/reprocess`, {
            method: 'POST',
        });
        
        if (!response.ok) {
            throw new Error('Reprocess failed');
        }
        
        await loadDocuments();
        hideLoading();
        showToast('Document reprocessed successfully', 'success');
    } catch (error) {
        hideLoading();
        showToast('Failed to reprocess document', 'error');
        console.error(error);
    }
}

async function deleteDocument(docId) {
    if (!confirm('Are you sure you want to delete this document?')) return;
    
    try {
        showLoading('Deleting document...');
        const response = await fetch(`${API_BASE}/documents/${docId}`, {
            method: 'DELETE',
        });
        
        if (!response.ok) {
            throw new Error('Delete failed');
        }
        
        await loadDocuments();
        hideLoading();
        showToast('Document deleted successfully', 'success');
    } catch (error) {
        hideLoading();
        showToast('Failed to delete document', 'error');
        console.error(error);
    }
}

async function toggleDocumentQuery(docId, currentState) {
    try {
        const nextState = !currentState;
        await apiCall(`/documents/${docId}/preferences`, 'PATCH', {
            query_enabled: nextState,
        });
        await loadDocuments();
        showToast(nextState ? 'Document enabled for queries' : 'Document excluded from queries', 'success');
    } catch (error) {
        showToast(error.message || 'Failed to update document', 'error');
        console.error(error);
    }
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

    // For assistant messages, add thinking section first (collapsed)
    let thinkingContainer = null;
    if (role === 'assistant') {
        thinkingContainer = document.createElement('details');
        thinkingContainer.className = 'thinking-container';
        
        const thinkingSummary = document.createElement('summary');
        thinkingSummary.textContent = '🧠 Thinking...';
        thinkingContainer.appendChild(thinkingSummary);
        
        const thinkingContent = document.createElement('div');
        thinkingContent.className = 'thinking-content';
        thinkingContainer.appendChild(thinkingContent);
        
        messageDiv.appendChild(thinkingContainer);
    }

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

    return { messageDiv, contentDiv, sourcesContainer, timeDiv, thinkingContainer };
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
        let message = 'Streaming request failed';
        try {
            const errorPayload = await response.json();
            if (errorPayload?.detail) {
                message = errorPayload.detail;
            }
        } catch (parseError) {
            // ignore JSON parse errors and fall back to default message
        }
        const error = new Error(message);
        error.status = response.status;
        if (message && message.toLowerCase().includes('no active documents selected')) {
            error.isNoDocuments = true;
        }
        throw error;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let accumulated = '';
    let thinkingSteps = [];

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

            if (payload.type === 'thinking') {
                // Handle thinking step
                const step = payload.step;
                thinkingSteps.push(step);
                updateThinkingDisplay(assistantElements, thinkingSteps);
                scrollToBottom();
            } else if (payload.type === 'chunk') {
                if (payload.content) {
                    accumulated += payload.content;
                    assistantElements.contentDiv.innerHTML = renderMarkdown(accumulated);
                    scrollToBottom();
                }
            } else if (payload.type === 'end') {
                accumulated = payload.content || accumulated;
                assistantElements.contentDiv.innerHTML = renderMarkdown(accumulated);
                // Finalize thinking display
                finalizeThinkingDisplay(assistantElements, thinkingSteps);
                if (payload.sources && payload.sources.length > 0) {
                    if (assistantElements.sourcesContainer) {
                        assistantElements.sourcesContainer.remove();
                    }
                    assistantElements.sourcesContainer = buildSourcesElement(payload.sources);
                    assistantElements.messageDiv.appendChild(assistantElements.sourcesContainer);
                }
                scrollToBottom();
            } else if (payload.type === 'error') {
                const error = new Error(payload.message || 'Streaming error');
                if (payload.message && payload.message.toLowerCase().includes('no active documents selected')) {
                    error.isNoDocuments = true;
                }
                throw error;
            }
        }
    }
}

function updateThinkingDisplay(assistantElements, thinkingSteps) {
    if (!assistantElements.thinkingContainer) return;
    
    const thinkingContent = assistantElements.thinkingContainer.querySelector('.thinking-content');
    if (!thinkingContent) return;
    
    // Update summary to show current step
    const summary = assistantElements.thinkingContainer.querySelector('summary');
    const lastStep = thinkingSteps[thinkingSteps.length - 1];
    if (lastStep && summary) {
        summary.textContent = `🧠 ${lastStep.message}`;
    }
    
    // Build thinking log
    let html = '<ul class="thinking-steps">';
    thinkingSteps.forEach((step, index) => {
        const icon = getThinkingIcon(step.type);
        html += `<li class="thinking-step ${step.type}">`;
        html += `<span class="step-icon">${icon}</span>`;
        html += `<span class="step-message">${escapeHtml(step.message)}</span>`;
        
        // Show details if available
        if (step.details) {
            if (Array.isArray(step.details)) {
                html += '<ul class="step-details">';
                step.details.forEach(detail => {
                    if (typeof detail === 'string') {
                        html += `<li>${escapeHtml(detail)}</li>`;
                    } else if (detail.text && detail.score !== undefined) {
                        html += `<li><span class="score">[${detail.score.toFixed(3)}]</span> ${escapeHtml(detail.text)}</li>`;
                    } else {
                        html += `<li>${escapeHtml(JSON.stringify(detail))}</li>`;
                    }
                });
                html += '</ul>';
            }
        }
        
        html += '</li>';
    });
    html += '</ul>';
    
    thinkingContent.innerHTML = html;
}

function finalizeThinkingDisplay(assistantElements, thinkingSteps) {
    if (!assistantElements.thinkingContainer) return;
    
    const summary = assistantElements.thinkingContainer.querySelector('summary');
    if (summary) {
        const stepCount = thinkingSteps.length;
        summary.textContent = `🧠 Thinking (${stepCount} steps) - click to expand`;
    }
    
    // Collapse the thinking section after completion
    assistantElements.thinkingContainer.removeAttribute('open');
}

function getThinkingIcon(stepType) {
    const icons = {
        'start': '🚀',
        'generating_queries': '✍️',
        'queries_generated': '📝',
        'searching': '🔍',
        'search_complete': '✅',
        'deduplication': '🔄',
        'reranking': '⚖️',
        'rerank_complete': '📊',
        'loading_parents': '📂',
        'complete': '✅',
        'no_results': '❌',
        'low_score': '⚠️',
        'retry_start': '🔁',
        'retry_queries_generated': '📝',
        'retry_searching': '🔍',
        'retry_deduplication': '🔄',
        'retry_reranking': '⚖️',
        'retry_rerank_complete': '📊',
        'retry_loading_parents': '📂',
        'retry_complete': '✅',
        'retry_no_results': '❌'
    };
    return icons[stepType] || '•';
}
