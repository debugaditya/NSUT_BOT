import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import {
  Loader2, LogOut, Github, BrainCircuit, Bot, User, Menu, Plus, 
  AlertCircle, CheckCircle, FileText, Image as ImageIcon // Added ImageIcon
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import 'katex/dist/katex.min.css';
import './style.css';

const API_BASE = import.meta.env.VITE_SERVER_URL || 'http://localhost:3000';

function DashboardPage() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [status, setStatus] = useState({ message: '', type: '' });

  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const textareaRef = useRef(null);
  const [file, setFile] = useState(null);

  useEffect(() => {
    verifySession();
  }, []);

  useEffect(() => {
    if (!loading) messagesEndRef.current?.scrollIntoView({ behavior: "auto" });
  }, [loading]);

  useEffect(() => {
    if (sending) messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, sending]);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      if (input === '') {
        textareaRef.current.style.height = '24px';
      } else {
        textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
      }
    }
  }, [input]);

  const showStatus = (msg, type) => {
    setStatus({ message: msg, type: type });
    setTimeout(() => setStatus({ message: '', type: '' }), 4000);
  };

  const preprocessLaTeX = (content) => {
    if (!content) return "";
    return content
      .replace(/\\\[/g, '$$')
      .replace(/\\\]/g, '$$')
      .replace(/\\\(/g, '$')
      .replace(/\\\)/g, '$');
  };

  async function verifySession() {
    try {
      const response = await fetch(`${API_BASE}/verify`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
      });
      if (response.status !== 200) throw new Error('Unauthorized');
      const data = await response.json();
      setMessages(data.chats || []);
      setLoading(false);
    } catch {
      navigate('/');
    }
  }

  // --- MODIFIED: Restrict to Images Only ---
  function handleFileSelect(e) {
    if (e.target.files[0]) {
        const selectedFile = e.target.files[0];
        
        // Strict Check: Must be an image
        if (!selectedFile.type.startsWith('image/')) {
            showStatus("Only image files (JPG, PNG, WEBP) are allowed.", "error");
            e.target.value = ""; // Reset input
            return;
        }

        setFile(selectedFile);
        setStatus({ message: '', type: '' });
    }
  }

  // --- Handle Paste (Ctrl+V) ---
  const handlePaste = (e) => {
    const items = e.clipboardData.items;
    for (let i = 0; i < items.length; i++) {
      if (items[i].type.indexOf('image') !== -1) {
        e.preventDefault(); 
        const blob = items[i].getAsFile();
        const file = new File([blob], "screenshot.png", { type: blob.type });
        setFile(file);
        setStatus({ message: 'Screenshot attached!', type: 'success' });
        setTimeout(() => setStatus({ message: '', type: '' }), 3000);
        return; 
      }
    }
  };

  async function deleteCookies() {
    try {
      await fetch(`${API_BASE}/logout`, { method: 'GET', credentials: 'include' });
      navigate('/');
    } catch (error) { console.error(error); }
  }

  async function handleSend() {
    if ((!input.trim() && !file) || sending) return;

    const currentInput = input;
    const currentFile = file;

    setInput('');
    setFile(null);
    setSending(true);
    setStatus({ message: '', type: '' });

    setMessages(prev => [...prev, { role: 'user', content: currentInput, document: currentFile ? currentFile.name : null }]);

    try {
      const formData = new FormData();
      formData.append("message", currentInput);
      if (currentFile) formData.append("file", currentFile);

      const response = await fetch(`${API_BASE}/send`, {
        method: 'POST',
        credentials: 'include',
        body: formData,
      });

      if (response.status === 413) {
        showStatus("File too large. Maximum size is 10MB.", "error");
        setSending(false); 
        return; 
      }
      
      if (!response.ok) throw new Error("Network response was not ok");
      if (!response.body) throw new Error("No response body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let botContent = '';
      let isFirstChunk = true;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        botContent += chunk;

        if (isFirstChunk) {
          isFirstChunk = false;
          setMessages(prev => [...prev, { role: 'model', content: botContent }]);
        } else {
          setMessages(prev => {
            const updated = [...prev];
            const lastMsg = updated[updated.length - 1];
            if (lastMsg && lastMsg.role === 'model') {
              lastMsg.content = botContent;
            }
            return updated;
          });
        }
      }
    } catch (error) {
      console.error("Streaming error:", error);
      if (status.type !== 'error') {
          setMessages(prev => [...prev, { role: 'model', content: '**Error:** Connection failed.' }]);
      }
    } finally {
      setSending(false);
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const MarkdownComponents = {
    code({ inline, className, children, ...props }) {
      const match = /language-(\w+)/.exec(className || '');
      return !inline && match ? (
        <div className="code-block-wrapper">
          <div className="code-header"><span>{match[1]}</span></div>
          <SyntaxHighlighter style={vscDarkPlus} language={match[1]} PreTag="div" {...props}>
            {String(children).replace(/\n$/, '')}
          </SyntaxHighlighter>
        </div>
      ) : (
        <code className={className} {...props}>{children}</code>
      );
    },
    img: (props) => <img className="msg-image" {...props} alt="Context" />,
    table: ({ children }) => <div className="table-container"><table>{children}</table></div>,
    p: ({ children }) => <p style={{ margin: "0.5rem 0" }}>{children}</p>
  };

  if (loading) return (
    <div className="loading-screen">
      <Loader2 className="spinner-large" size={48} />
      <p>Authenticating NSUT Bot...</p>
    </div>
  );

  return (
    <div className="dashboard-layout">
      {/* Mobile Sidebar Toggle */}
      <button className="mobile-toggle" onClick={() => setSidebarOpen(!sidebarOpen)}>
        <Menu size={20} />
      </button>

      {/* Sidebar */}
      <AnimatePresence mode='wait'>
        {sidebarOpen && (
          <motion.aside
            className="sidebar"
            initial={{ x: -250 }} animate={{ x: 0 }} exit={{ x: -250 }}
            transition={{ duration: 0.2 }}
          >
            <div className="sidebar-header">
              <BrainCircuit size={28} className="accent-text" />
              <h2>NSUT<span className="accent-text">Bot</span></h2>
            </div>
            <nav className="sidebar-nav">
              <button onClick={() => navigate('/train')} className="nav-item">
                <BrainCircuit size={20} /><span>Train the model</span>
              </button>
              <button onClick={() => window.open('https://github.com/debugaditya', '_blank')} className="nav-item">
                <Github size={20} /><span>Developer</span>
              </button>
            </nav>
            <div className="sidebar-footer">
              <div className="user-mini-profile">
                <div className="status-dot"></div><span>Connected</span>
              </div>
              <button onClick={deleteCookies} className="nav-item logout">
                <LogOut size={20} /><span>Logout</span>
              </button>
            </div>
          </motion.aside>
        )}
      </AnimatePresence>

      {/* Main Chat Area */}
      <main className="chat-area">

        <div className="messages-scroll-area">
          {messages.length === 0 ? (
            <div className="empty-state">
              <Bot size={64} className="empty-icon" />
              <h3>How can I help you today?</h3>
              <p>Ask about coursework, syllabus, or coding problems.</p>
            </div>
          ) : (
            <>
              {messages.map((msg, i) => (
                <div key={i} className={`message-wrapper ${msg.role}`}>
                  <div className="avatar">
                    {msg.role === 'user' ? <User size={20} /> : <Bot size={30} />}
                  </div>

                  <div className="message-bubble">
                    {msg.document && (
                      <div className="file-attachment" style={{
                        display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px',
                        padding: '8px', background: 'rgba(0, 0, 0, 0.2)', borderRadius: '6px', fontSize: '0.85rem'
                      }}>
                        {/* Use Image Icon instead of FileText for images */}
                        <ImageIcon size={16} />
                        <span style={{ fontWeight: 500 }}>{msg.document}</span>
                      </div>
                    )}

                    <ReactMarkdown
                      remarkPlugins={[remarkMath]}
                      rehypePlugins={[rehypeKatex]}
                      components={MarkdownComponents}
                    >
                      {preprocessLaTeX(msg.content)}
                    </ReactMarkdown>
                  </div>
                </div>
              ))}

              {sending && messages[messages.length - 1]?.role === 'user' && (
                <div className="message-wrapper model">
                  <div className="avatar"><Bot size={30} /></div>
                  <div className="message-bubble">
                    <div className="typing-indicator">
                      <div className="typing-dot"></div>
                      <div className="typing-dot"></div>
                      <div className="typing-dot"></div>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Region */}
        <div className="input-area">
          
          <AnimatePresence>
            {status.message && (
              <motion.div 
                initial={{ opacity: 0, y: 10, height: 0 }}
                animate={{ opacity: 1, y: 0, height: 'auto' }}
                exit={{ opacity: 0, y: 10, height: 0 }}
                style={{ marginBottom: '10px' }}
                className={`status-message ${status.type}`}
              >
                {status.type === 'success' ? <CheckCircle size={16} /> : <AlertCircle size={16} />}
                <span>{status.message}</span>
              </motion.div>
            )}
          </AnimatePresence>

          {file && (
            <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '0.5rem' }}>
              <div style={{
                display: "inline-flex", alignItems: "center", gap: "8px",
                background: "#1e293b", border: "1px solid rgba(255,255,255,0.1)",
                padding: "6px 12px", borderRadius: "99px", fontSize: "0.85rem"
              }}>
                <span>📷 {file.name}</span>
                <button
                  onClick={() => setFile(null)}
                  style={{ background: "none", border: "none", color: "#94a3b8", cursor: "pointer", marginLeft: "4px" }}
                >✕</button>
              </div>
            </div>
          )}

          <div className="input-container">
            <input
              type="file"
              ref={fileInputRef}
              style={{ display: "none" }}
              onChange={handleFileSelect}
              accept="image/*" // <--- RESTRICT SYSTEM PICKER TO IMAGES ONLY
            />

            <button
              className="attach-btn"
              type="button"
              onClick={() => fileInputRef.current.click()}
              title="Attach Image"
            >
              <Plus size={20} />
            </button>

            <textarea
              ref={textareaRef}
              className="chat-input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onPaste={handlePaste}
              placeholder="Ask a question (Ctrl+V to paste screenshot)"
              onKeyDown={handleKeyDown}
              disabled={sending}
              autoFocus
              rows={1}
            />

            <button
              className="send-btn"
              onClick={handleSend}
              disabled={sending || (!input.trim() && !file)}
            >
              {sending ? <div className="css-spinner"></div> : <span className="send-text">➤</span>}
            </button>
          </div>
          <div className="disclaimer">NSUT Bot can make mistakes. Check important info.</div>
        </div>
      </main>
    </div>
  );
}

export default DashboardPage;