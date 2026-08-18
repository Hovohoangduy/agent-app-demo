const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const chatMessages = document.getElementById('chat-messages');
const sendBtn = document.getElementById('send-btn');

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const text = userInput.value.trim();
    if (!text) return;
    
    // Add user message to UI
    addMessage(text, 'user');
    userInput.value = '';
    
    // Disable input while generating
    userInput.disabled = true;
    sendBtn.disabled = true;
    
    // Show typing indicator
    const typingId = showTypingIndicator();
    
    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query: text })
        });
        
        if (!response.ok) throw new Error('Network response was not ok');
        
        removeTypingIndicator(typingId);
        const contentDiv = addMessage('', 'assistant');
        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let currentText = '';
        let buffer = '';
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Keep incomplete line in buffer
            
            for (const line of lines) {
                const trimmed = line.trim();
                if (trimmed.startsWith('data: ')) {
                    const dataStr = trimmed.slice(6);
                    if (dataStr === '[DONE]') break;
                    
                    try {
                        const data = JSON.parse(dataStr);
                        if (data.content) {
                            currentText += data.content;
                            if (typeof marked !== 'undefined') {
                                contentDiv.innerHTML = marked.parse(currentText);
                            } else {
                                contentDiv.textContent = currentText;
                            }
                            scrollToBottom();
                        } else if (data.error) {
                            currentText += "\n[Error: " + data.error + "]";
                            contentDiv.textContent = currentText;
                            scrollToBottom();
                        }
                    } catch(e) {
                        console.error("Error parsing JSON chunk:", e, dataStr);
                    }
                }
            }
        }
        
    } catch (error) {
        console.error('Error:', error);
        removeTypingIndicator(typingId);
        addMessage("Sorry, I encountered an error while trying to process your request.", 'assistant');
    } finally {
        // Re-enable input
        userInput.disabled = false;
        sendBtn.disabled = false;
        userInput.focus();
    }
});

function addMessage(text, sender) {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message', sender);
    
    const contentDiv = document.createElement('div');
    contentDiv.classList.add('message-content');
    contentDiv.textContent = text; // Prevent XSS by using textContent
    
    msgDiv.appendChild(contentDiv);
    chatMessages.appendChild(msgDiv);
    
    scrollToBottom();
    return contentDiv;
}

function showTypingIndicator() {
    const id = 'typing-' + Date.now();
    const typingDiv = document.createElement('div');
    typingDiv.classList.add('typing-indicator');
    typingDiv.id = id;
    
    for (let i = 0; i < 3; i++) {
        const dot = document.createElement('div');
        dot.classList.add('dot');
        typingDiv.appendChild(dot);
    }
    
    chatMessages.appendChild(typingDiv);
    scrollToBottom();
    return id;
}

function removeTypingIndicator(id) {
    const indicator = document.getElementById(id);
    if (indicator) {
        indicator.remove();
    }
}

function scrollToBottom() {
    const container = document.querySelector('.chat-container');
    container.scrollTop = container.scrollHeight;
}
