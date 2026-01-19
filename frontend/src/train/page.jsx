import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  UploadCloud, 
  CheckCircle, 
  AlertCircle, 
  ArrowLeft, 
  Loader2, 
  Database,
  BrainCircuit
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import './style.css';

const API_BASE = import.meta.env.VITE_SERVER_URL || 'http://localhost:3000';

function Train() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [user, setUser] = useState({ name: '', email: '' });
  const [status, setStatus] = useState({ type: '', message: '' });
  const [selectedFile, setSelectedFile] = useState(null);

  useEffect(() => {
    verifySession();
  }, []);

  async function verifySession() {
    try {
      const response = await fetch(`${API_BASE}/verify`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
      });
      
      if (response.status !== 200) throw new Error('Unauthorized');
      
      const data = await response.json();
      setUser({ name: data.name, email: data.email });
      setLoading(false);
    } catch (error) {
      navigate('/');
    }
  }

  async function handleUpload(e) {
    e.preventDefault();
    if (!selectedFile) return;

    setUploading(true);
    setStatus({ type: '', message: '' });

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(`${API_BASE}/train`, {
        method: 'POST',
        body: formData, 
        credentials: 'include',
      });

      if (response.status === 200) {
        const result = await response.json();
        setStatus({ 
          type: 'success', 
          message: `Success! Added ${result.chunks_generated} new chunks.` 
        });
        setSelectedFile(null); 
      } else {
        throw new Error('Upload failed on server.');
      }
    } catch (error) {
      setStatus({ type: 'error', message: 'Upload failed. Try again.' });
      console.error(error);
    } finally {
      setUploading(false);
    }
  }

  if (loading) {
    return (
      <div className="loading-screen">
        <Loader2 className="spinner-large" size={64} />
      </div>
    );
  }

  return (
    <div className="dashboard-layout">
      {/* Navbar */}
      <nav className="train-nav">
        <button onClick={() => navigate('/dashboard')} className="back-btn">
          <ArrowLeft size={18} /> <span className="hide-mobile">Back</span>
        </button>
        <div className="user-badge">
          <div className="status-dot"></div>
          <span className="user-email">{user.email}</span>
        </div>
      </nav>

      <main className="train-container">
        <motion.div 
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="train-card compact"
        >
          {/* Centered Header */}
          <div className="train-header">
            <div className="icon-wrapper">
              <Database size={28} />
            </div>
            <div>
              <h1>Knowledge Base</h1>
              <p className="subtitle">Train the NSUT Bot</p>
            </div>
          </div>

          {/* Full Explanation Paragraph */}
          <div className="motivation-section">
            <h3>Why Upload Data?</h3>
            <p>
              This chatbot relies on <strong>Retrieval-Augmented Generation (RAG)</strong>. 
              Unlike standard AI that "hallucinates", our bot needs real source material.
            </p>
            <p>
              By uploading <strong>Notes, Syllabi, Past Papers, and Lab Manuals</strong>, 
              you are expanding the brain of this AI. Every document helps it answer questions 
              more accurately for the entire NSUT community.
            </p>
          </div>

          <form onSubmit={handleUpload} className="upload-form">
            <div className={`drop-zone ${selectedFile ? 'has-file' : ''}`}>
              <input 
                type="file" 
                id="file-upload" 
                onChange={(e) => setSelectedFile(e.target.files[0])}
                required 
                accept=".pdf,.docx,.pptx,.txt,.md,.csv,.py,.js,.json"
              />
              <label htmlFor="file-upload">
                <UploadCloud size={32} className="upload-icon" />
                <div className="file-info">
                  {selectedFile ? (
                    <span className="file-name">{selectedFile.name}</span>
                  ) : (
                    <span>Tap to select file</span>
                  )}
                </div>
              </label>
            </div>

            <div className="supported-formats">
              <span className="label">Supported:</span>
              <div className="formats-list">
                <span className="tag">PDF</span>
                <span className="tag">DOCX</span>
                <span className="tag">PPTX</span>
                <span className="tag">TXT</span>
                <span className="tag">CODE</span>
              </div>
            </div>

            <button 
              type="submit" 
              className="upload-btn" 
              disabled={uploading || !selectedFile}
            >
              {uploading ? <Loader2 className="spinner" size={18} /> : <BrainCircuit size={18} />}
              <span>{uploading ? 'Processing...' : 'Train Model'}</span>
            </button>
          </form>

          <AnimatePresence>
            {status.message && (
              <motion.div 
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                className={`status-message ${status.type}`}
              >
                {status.type === 'success' ? <CheckCircle size={16} /> : <AlertCircle size={16} />}
                <span>{status.message}</span>
              </motion.div>
            )}
          </AnimatePresence>

        </motion.div>
      </main>
    </div>
  );
}

export default Train;