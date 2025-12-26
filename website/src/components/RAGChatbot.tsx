import React, { useState, useEffect, useRef, lazy, Suspense } from 'react';
import ReactMarkdown from 'react-markdown';
import '@site/src/css/custom.css';
import '@site/src/css/voice-chat.css';
import '@site/src/css/voice-chat-kids.css';

// Lazy load VoiceChat components for better performance
const VoiceChat = lazy(() => import('./VoiceChat'));
const VoiceChatKids = lazy(() => import('./VoiceChatKids'));

interface Message {
  role: 'user' | 'assistant';
  content: string;
  context?: string;
}

// Use local backend in development, Railway in production
const BACKEND_URL = process.env.NODE_ENV === 'development'
  ? 'http://localhost:8000'
  : 'https://physical-ai-backend-production-1c69.up.railway.app';

export default function RAGChatbot(): React.ReactElement {
  const [isOpen, setIsOpen] = useState(false);
  const [isVoiceOpen, setIsVoiceOpen] = useState(false);
  const [useKidsMode, setUseKidsMode] = useState(true); // Default to kids-friendly mode
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'assistant',
      content: 'Hi! I\'m your AI Plant Biotechnology assistant. Ask me anything about ML in agriculture, computer vision for plants, genomics, or any content from the book! You can also select text on the page and ask me questions about it.'
    }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [selectedText, setSelectedText] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Listen for text selection
  useEffect(() => {
    const handleSelection = () => {
      const selection = window.getSelection();
      const text = selection?.toString().trim();
      if (text && text.length > 3) {
        setSelectedText(text);
        setIsOpen(true);
      }
    };

    document.addEventListener('mouseup', handleSelection);
    return () => document.removeEventListener('mouseup', handleSelection);
  }, []);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage: Message = {
      role: 'user',
      content: selectedText ? `Context: "${selectedText}"\n\nQuestion: ${input}` : input,
      context: selectedText || undefined
    };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      // Build conversation history for the chat endpoint
      const conversationHistory = [...messages, userMessage].map(msg => ({
        role: msg.role,
        content: msg.role === 'user' && msg.context
          ? `Context: "${msg.context}"\n\nQuestion: ${msg.content}`
          : msg.content
      }));

      const response = await fetch(`${BACKEND_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: conversationHistory
        })
      });

      if (!response.ok) {
        throw new Error('Failed to get response');
      }

      const data = await response.json();
      const botMessage: Message = { role: 'assistant', content: data.response };
      setMessages(prev => [...prev, botMessage]);
      setSelectedText(''); // Clear selected text after using it
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please make sure the backend server is running at ' + BACKEND_URL
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const openVoiceChat = () => {
    setIsVoiceOpen(true);
  };

  return (
    <>
      <div className="chatbot-container">
        {/* Floating buttons */}
        {!isOpen && (
          <div className="chatbot-fab-container">
            {/* Voice FAB */}
            <button
              className="chatbot-fab voice-fab"
              onClick={openVoiceChat}
              aria-label="Start voice chat"
              title="Talk to Plant AI"
            >
              <div className="voice-fab-pulse" />
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                <line x1="12" y1="19" x2="12" y2="23"/>
                <line x1="8" y1="23" x2="16" y2="23"/>
              </svg>
            </button>
            {/* Chat FAB */}
            <button
              className="chatbot-button"
              onClick={() => setIsOpen(true)}
              aria-label="Open chatbot"
            >
              💬
            </button>
          </div>
        )}

        {/* Chatbot window */}
        {isOpen && (
          <div className="chatbot-window">
            {/* Header */}
            <div className="chatbot-header">
              <span>🌱 Plant AI Assistant</span>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                {/* Voice Button in Header */}
                <button
                  className="chatbot-voice-btn"
                  onClick={openVoiceChat}
                  aria-label="Start voice chat"
                  title="Switch to voice mode"
                >
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                    <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                    <line x1="12" y1="19" x2="12" y2="23"/>
                    <line x1="8" y1="23" x2="16" y2="23"/>
                  </svg>
                </button>
                <button
                  className="chatbot-close"
                  onClick={() => setIsOpen(false)}
                  aria-label="Close chatbot"
                >
                  ✕
                </button>
              </div>
            </div>

            {/* Voice Mode Banner */}
            <div className="voice-mode-banner" onClick={openVoiceChat}>
              <div className="voice-mode-icon">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                  <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                </svg>
              </div>
              <div className="voice-mode-text">
                <span className="voice-mode-title">🎙️ Try Voice Mode!</span>
                <span className="voice-mode-subtitle">Talk naturally with Plant AI</span>
              </div>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="voice-mode-arrow">
                <path d="M9 18l6-6-6-6"/>
              </svg>
            </div>

            {/* Messages */}
            <div className="chatbot-messages">
              {messages.map((msg, idx) => (
                <div key={idx}>
                  {msg.context && msg.role === 'user' && (
                    <div className="context-bubble">
                      <div className="context-label">📄 Context</div>
                      <div className="context-text">{msg.context}</div>
                    </div>
                  )}
                  <div className={`message ${msg.role === 'user' ? 'message-user' : 'message-bot'}`}>
                    {msg.role === 'assistant' ? (
                      <ReactMarkdown>{msg.content}</ReactMarkdown>
                    ) : (
                      msg.content
                    )}
                  </div>
                </div>
              ))}
              {loading && (
                <div className="message message-bot loading-message">
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div className="chatbot-input-container">
              {selectedText && (
                <div className="selected-text-preview">
                  <div className="selected-text-header">
                    <span>📄 Selected Context</span>
                    <button 
                      className="clear-context-btn"
                      onClick={() => setSelectedText('')}
                      title="Clear context"
                    >
                      ✕
                    </button>
                  </div>
                  <div className="selected-text-content">
                    {selectedText.length > 100 ? selectedText.substring(0, 100) + '...' : selectedText}
                  </div>
                </div>
              )}
              <div className="input-wrapper">
                <input
                  type="text"
                  className="chatbot-input"
                  placeholder="Ask a question..."
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyPress={handleKeyPress}
                  disabled={loading}
                />
                <button
                  className="chatbot-send"
                  onClick={sendMessage}
                  disabled={loading || !input.trim()}
                  title="Send message"
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="22" y1="2" x2="11" y2="13"></line>
                    <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                  </svg>
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Voice Chat Modal - Kids-friendly version */}
      {isVoiceOpen && (
        <Suspense fallback={
          <div className="voice-chat-overlay">
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
              <div className="voice-loading-spinner" style={{ width: 48, height: 48 }} />
            </div>
          </div>
        }>
          {useKidsMode ? (
            <VoiceChatKids isOpen={isVoiceOpen} onClose={() => setIsVoiceOpen(false)} />
          ) : (
            <VoiceChat isOpen={isVoiceOpen} onClose={() => setIsVoiceOpen(false)} />
          )}
        </Suspense>
      )}
    </>
  );
}
