const API_BASE = window.location.hostname === 'localhost'
    ? 'http://localhost:8000'
    : '/api';

const ErrorMessages = {
    CHAT_LOAD: 'Failed to load chats',
    CHAT_CREATE: 'Failed to create chat',
    CHAT_DELETE: 'Failed to delete chat',
    MESSAGES_LOAD: 'Failed to load messages',
    DOCUMENTS_LOAD: 'Failed to load documents',
    DOCUMENT_UPLOAD: 'Failed to upload document',
    DOCUMENT_DELETE: 'Failed to delete document',
    DOCUMENT_REPROCESS: 'Failed to reprocess document',
    DOCUMENT_UPDATE: 'Failed to update document',
    QUERY_PREPARE: 'Failed to prepare chat',
    QUERY_RESPONSE: 'Failed to get response',
    NO_DOCUMENTS: 'No active documents selected for querying',
    NETWORK: 'Network error - please check your connection',
    SERVER: 'Server error - please try again later'
};

let currentChatId = null;
let chats = [];
let documents = [];

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

document.addEventListener('DOMContentLoaded', () => {
    Promise.all([loadChats(), loadDocuments()]).catch(console.error);
    setupEventListeners();
});

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
            sendQuery().then();
        }
    });
}

class ApiError extends Error {
    constructor(message, status, detail) {
        super(message);
        this.status = status;
        this.detail = detail;
    }
}

async function apiCall(endpoint, method = 'GET', data = null) {
    const options = {
        method,
        headers: { 'Content-Type': 'application/json' },
    };

    if (data) {
        options.body = JSON.stringify(data);
    }

    let response;
    try {
        response = await fetch(`${API_BASE}${endpoint}`, options);
    } catch (networkError) {
        throw new ApiError(ErrorMessages.NETWORK, 0, networkError.message);
    }

    if (!response.ok) {
        let detail = 'Request failed';
        try {
            const errorBody = await response.json();
            detail = errorBody.detail || detail;
        } catch (parseError) {}

        if (response.status >= 500) {
            throw new ApiError(ErrorMessages.SERVER, response.status, detail);
        }
        throw new ApiError(detail, response.status, detail);
    }

    return response.json();
}

async function loadChats() {
    try {
        chats = await apiCall('/chats');
        renderChats();

        if (!currentChatId && chats.length > 0) {
            await selectChat(chats[0].id);
        }
    } catch (error) {
        showToast(`${ErrorMessages.CHAT_LOAD}: ${error.detail || error.message}`, 'error');
        console.error(error);
    }
}

function renderChats() {
    chatList.innerHTML = '';

    if (chats.length === 0) {
        chatList.innerHTML = '<div class="empty-state">No chats yet</div>';
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
            <div class="chat-item-title">${escapeHtml(chat.title)}</div>
            <div class="chat-item-date">${dateStr}</div>
        `;

        chatItem.addEventListener('click', () => selectChat(chat.id));
        chatList.appendChild(chatItem);
    });
}

async function createNewChat() {
    const title = prompt('Enter chat title:', 'New Chat');
    if (!title) return;

    try {
        showLoading('Creating chat...');
        const chat = await apiCall('/chats', 'POST', { title });
        chats.unshift(chat);
        renderChats();
        await selectChat(chat.id);
        showToast('Chat created successfully', 'success');
    } catch (error) {
        showToast(`${ErrorMessages.CHAT_CREATE}: ${error.detail || error.message}`, 'error');
        console.error(error);
    } finally {
        hideLoading();
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
    if (currentChatId) return currentChatId;

    if (chats.length > 0) {
        await selectChat(chats[0].id);
        return currentChatId;
    }

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
    } catch (error) {
        showToast(`${ErrorMessages.MESSAGES_LOAD}: ${error.detail || error.message}`, 'error');
        console.error(error);
    } finally {
        hideLoading();
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

    try {
        await ensureActiveChat();
    } catch (error) {
        showToast(`${ErrorMessages.QUERY_PREPARE}: ${error.detail || error.message}`, 'error');
        console.error(error);
        return;
    }

    const query = queryInput.value.trim();
    queryInput.value = '';
    queryInput.disabled = true;
    sendBtn.disabled = true;

    addMessageToUI('user', query);

    try {
        const assistantElements = addMessageToUI('assistant', '');
        await streamQuery(query, assistantElements);
    } catch (error) {
        if (error.detail?.toLowerCase().includes('no active documents')) {
            addAssistantInfoMessage('Please enable at least one document in the right panel before asking questions.');
        } else {
            showToast(`${ErrorMessages.QUERY_RESPONSE}: ${error.detail || error.message}`, 'error');
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
        showToast('Chat deleted successfully', 'success');
    } catch (error) {
        showToast(`${ErrorMessages.CHAT_DELETE}: ${error.detail || error.message}`, 'error');
        console.error(error);
    } finally {
        hideLoading();
    }
}

async function loadDocuments() {
    try {
        documents = await apiCall('/documents');
        renderDocuments();
    } catch (error) {
        showToast(`${ErrorMessages.DOCUMENTS_LOAD}: ${error.detail || error.message}`, 'error');
        console.error(error);
    }
}

function renderDocuments() {
    documentsList.innerHTML = '';

    if (documents.length === 0) {
        documentsList.innerHTML = '<div class="empty-state">No documents uploaded</div>';
        return;
    }

    documents.forEach(doc => {
        const docItem = document.createElement('div');
        docItem.className = 'document-item';

        const date = new Date(doc.uploaded_at).toLocaleDateString();
        const isProcessed = doc.processed;
        const status = isProcessed ? 'processed' : 'unprocessed';
        const statusText = isProcessed ? `${doc.num_chunks} chunks` : 'Not processed';
        const queryPillClass = doc.query_enabled ? 'query-pill active' : 'query-pill inactive';

        docItem.innerHTML = `
            <div class="document-name">${escapeHtml(doc.filename)}</div>
            <div class="document-info-row">
                <span>${date}</span>
                <span class="document-collection">${escapeHtml(doc.collection_name)}</span>
            </div>
            <div class="document-status-row">
                <div class="document-status ${status}">${statusText}</div>
                <div class="${queryPillClass}">${doc.query_enabled ? 'Query on' : 'Query off'}</div>
            </div>
            <div class="document-actions">
                ${!isProcessed ? `<button class="btn-reprocess" data-id="${doc.id}">Reprocess</button>` : ''}
                <button class="doc-toggle ${doc.query_enabled ? 'active' : ''}" data-id="${doc.id}" aria-pressed="${doc.query_enabled}">${doc.query_enabled ? 'ON' : 'OFF'}</button>
                <button class="btn-delete-doc" data-id="${doc.id}">Delete</button>
            </div>
        `;

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
        await apiCall(`/documents/${docId}/reprocess`, 'POST');
        await loadDocuments();
        showToast('Document reprocessed successfully', 'success');
    } catch (error) {
        showToast(`${ErrorMessages.DOCUMENT_REPROCESS}: ${error.detail || error.message}`, 'error');
        console.error(error);
    } finally {
        hideLoading();
    }
}

async function deleteDocument(docId) {
    if (!confirm('Are you sure you want to delete this document?')) return;

    try {
        showLoading('Deleting document...');
        await apiCall(`/documents/${docId}`, 'DELETE');
        await loadDocuments();
        showToast('Document deleted successfully', 'success');
    } catch (error) {
        showToast(`${ErrorMessages.DOCUMENT_DELETE}: ${error.detail || error.message}`, 'error');
        console.error(error);
    } finally {
        hideLoading();
    }
}

async function toggleDocumentQuery(docId, currentState) {
    try {
        const nextState = !currentState;
        await apiCall(`/documents/${docId}/preferences`, 'PATCH', { query_enabled: nextState });
        await loadDocuments();
        showToast(nextState ? 'Document enabled for queries' : 'Document excluded from queries', 'success');
    } catch (error) {
        showToast(`${ErrorMessages.DOCUMENT_UPDATE}: ${error.detail || error.message}`, 'error');
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
            let detail = 'Upload failed';
            try {
                const errorBody = await response.json();
                detail = errorBody.detail || detail;
            } catch (parseError) {}
            throw new ApiError(detail, response.status, detail);
        }

        const doc = await response.json();
        documents.unshift(doc);
        renderDocuments();

        showToast('Document uploaded successfully', 'success');
    } catch (error) {
        showToast(`${ErrorMessages.DOCUMENT_UPLOAD}: ${error.detail || error.message}`, 'error');
        console.error(error);
    } finally {
        hideLoading();
        fileInput.value = '';
    }
}

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
    }, 4000);
}

function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function throttle(func, delay) {
    let timeoutId;
    let lastRan;
    return function(...args) {
        if (!lastRan) {
            func.apply(this, args);
            lastRan = Date.now();
        } else {
            clearTimeout(timeoutId);
            timeoutId = setTimeout(() => {
                if ((Date.now() - lastRan) >= delay) {
                    func.apply(this, args);
                    lastRan = Date.now();
                }
            }, delay - (Date.now() - lastRan));
        }
    };
}

const throttledScroll = throttle(scrollToBottom, 100);

function createMessageElement(role, content, options = {}) {
    const { sources = null, timestamp = null } = options;
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    let thinkingContainer = null;
    if (role === 'assistant') {
        thinkingContainer = document.createElement('details');
        thinkingContainer.className = 'thinking-container';

        const thinkingSummary = document.createElement('summary');
        thinkingSummary.textContent = 'Thinking...';
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
    if (!text) return '';
    const html = marked.parse(text, { breaks: true });
    return DOMPurify.sanitize(html);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

async function streamQuery(query, assistantElements) {
    let response;
    try {
        response = await fetch(`${API_BASE}/query/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ chat_id: currentChatId, query }),
        });
    } catch (networkError) {
        throw new ApiError(ErrorMessages.NETWORK, 0, networkError.message);
    }

    if (!response.ok || !response.body) {
        let detail = 'Streaming request failed';
        try {
            const errorPayload = await response.json();
            detail = errorPayload.detail || detail;
        } catch (parseError) {}
        throw new ApiError(detail, response.status, detail);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let accumulated = '';
    let thinkingSteps = [];

    while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        let separatorIndex;
        while ((separatorIndex = buffer.indexOf('\n\n')) !== -1) {
            const rawEvent = buffer.slice(0, separatorIndex).trim();
            buffer = buffer.slice(separatorIndex + 2);

            if (!rawEvent.startsWith('data:')) continue;

            const dataString = rawEvent.replace(/^data:\s*/, '');
            if (!dataString) continue;

            let payload;
            try {
                payload = JSON.parse(dataString);
            } catch (err) {
                console.error('Failed to parse SSE payload', err);
                continue;
            }

            if (payload.type === 'thinking') {
                thinkingSteps.push(payload.step);
                updateThinkingDisplay(assistantElements, thinkingSteps);
                throttledScroll();
            } else if (payload.type === 'chunk') {
                if (payload.content) {
                    accumulated += payload.content;
                    assistantElements.contentDiv.textContent = accumulated;
                    throttledScroll();
                }
            } else if (payload.type === 'end') {
                accumulated = payload.content || accumulated;
                assistantElements.contentDiv.innerHTML = renderMarkdown(accumulated);
                finalizeThinkingDisplay(assistantElements, thinkingSteps);
                if (payload.sources?.length > 0) {
                    if (assistantElements.sourcesContainer) {
                        assistantElements.sourcesContainer.remove();
                    }
                    assistantElements.sourcesContainer = buildSourcesElement(payload.sources);
                    assistantElements.messageDiv.appendChild(assistantElements.sourcesContainer);
                }
                scrollToBottom();
            } else if (payload.type === 'error') {
                throw new ApiError(payload.message || 'Streaming error', 500, payload.message);
            }
        }
    }
}

function updateThinkingDisplay(assistantElements, thinkingSteps) {
    if (!assistantElements.thinkingContainer) return;

    const thinkingContent = assistantElements.thinkingContainer.querySelector('.thinking-content');
    if (!thinkingContent) return;

    const summary = assistantElements.thinkingContainer.querySelector('summary');
    const lastStep = thinkingSteps[thinkingSteps.length - 1];
    if (lastStep && summary) {
        summary.textContent = lastStep.message;
    }

    let stepsList = thinkingContent.querySelector('.thinking-steps');
    if (!stepsList) {
        stepsList = document.createElement('ul');
        stepsList.className = 'thinking-steps';
        thinkingContent.appendChild(stepsList);
    }

    const step = lastStep;
    const icon = getThinkingIcon(step.type);

    const li = document.createElement('li');
    li.className = `thinking-step ${step.type}`;

    const iconSpan = document.createElement('span');
    iconSpan.className = 'step-icon';
    iconSpan.textContent = icon;
    li.appendChild(iconSpan);

    const messageSpan = document.createElement('span');
    messageSpan.className = 'step-message';
    messageSpan.textContent = step.message;
    li.appendChild(messageSpan);

    if (step.details && Array.isArray(step.details)) {
        const detailsList = document.createElement('ul');
        detailsList.className = 'step-details';

        step.details.forEach(detail => {
            const detailLi = document.createElement('li');
            if (typeof detail === 'string') {
                detailLi.textContent = detail;
            } else if (detail.text && detail.score !== undefined) {
                const scoreSpan = document.createElement('span');
                scoreSpan.className = 'score';
                scoreSpan.textContent = `[${detail.score.toFixed(3)}]`;
                detailLi.appendChild(scoreSpan);
                detailLi.appendChild(document.createTextNode(` ${detail.text}`));
            } else {
                detailLi.textContent = JSON.stringify(detail);
            }
            detailsList.appendChild(detailLi);
        });

        li.appendChild(detailsList);
    }

    stepsList.appendChild(li);
}

function finalizeThinkingDisplay(assistantElements, thinkingSteps) {
    if (!assistantElements.thinkingContainer) return;

    const summary = assistantElements.thinkingContainer.querySelector('summary');
    if (summary) {
        summary.textContent = `Thinking (${thinkingSteps.length} steps) - click to expand`;
    }

    assistantElements.thinkingContainer.removeAttribute('open');
}

function getThinkingIcon(stepType) {
    const icons = {
        'start': '>',
        'round1_start': '1',
        'round2_start': '2',
        'round3_start': '3',
        'generating_queries': 'Q',
        'queries_generated': 'Q',
        'searching': 'S',
        'search_complete': 'S',
        'deduplication': 'D',
        'round1_dedup': 'D',
        'round2_dedup': 'D',
        'round3_dedup': 'D',
        'round1_reranking': 'R',
        'round2_reranking': 'R',
        'round3_reranking': 'R',
        'reranking': 'R',
        'rerank_complete': 'R',
        'round1_score': '#',
        'round2_score': '#',
        'round3_score': '#',
        'round1_success': '+',
        'round2_success': '+',
        'round1_acceptable': '+',
        'round2_final': '!',
        'round1_no_results': 'X',
        'no_results_final': 'X',
        'loading_parents': 'P',
        'complete': '*',
        'no_results': 'X',
        'no_documents': '!',
        'metadata_injection': 'M'
    };
    return icons[stepType] || '-';
}
