import { useState, useEffect } from 'react'
import './App.css'
import { GoogleLogin } from '@react-oauth/google'
import { useNavigate } from "react-router-dom";
import { motion } from 'framer-motion';
import { Cpu } from 'lucide-react'; // Only keeping the Logo 

const API_BASE = import.meta.env.VITE_SERVER_URL || 'http://localhost:3000';

function App() {
  // --- 🔒 LOGIC STARTS HERE (EXACTLY THE SAME) ---
  const [loading, setLoading] = useState(true);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const navigate = useNavigate();

  useEffect(() => {
    verifySession();
  }, []);

  const handleMouseMove = (e) => {
    setMousePosition({
      x: (e.clientX / window.innerWidth) * 20 - 10,
      y: (e.clientY / window.innerHeight) * 20 - 10,
    });
  };

  async function verifySession() {
    try {
      const response = await fetch(`${API_BASE}/verify`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include', 
      });

      if (response.ok) {
        navigate('/dashboard');
      } else {
        setLoading(false);
      }
    } catch (error) {
      console.error("Session verification failed:", error);
      setLoading(false);
    }
  }

  async function handleLogin(googleToken) {
    try {
      const response = await fetch(`${API_BASE}/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ token: googleToken }),
      });

      if (response.ok) {
        navigate('/dashboard');
      } else {
        const data = await response.json();
        if (response.status === 402) {
          alert(data.detail || 'Sign in with NSUT mail ID');
        } else {
          console.error('Login Failed:', data);
        }
      }
    } catch (error) {
      console.error('Login Network Error:', error);
    }
  }
  // --- 🔒 LOGIC ENDS HERE ---

  return (
    <>
      {loading ? (
        <div className="loading-container">
          <div className="spinner"></div>
        </div>
      ) : (
        <div
          className="app-container"
          onMouseMove={handleMouseMove}
          style={{ 
            background: '#0f172a',
            position: 'relative',
            overflow: 'hidden'
          }}
        >
          {/* --- Interactive Background Orbs --- */}
          <motion.div 
            className="orb"
            style={{
              position: 'absolute', top: '15%', left: '25%',
              width: '300px', height: '300px',
              background: 'radial-gradient(circle, rgba(139,92,246,0.25) 0%, rgba(0,0,0,0) 70%)',
              filter: 'blur(40px)', zIndex: 0,
              x: mousePosition.x * -1.5, y: mousePosition.y * -1.5
            }}
          />
          <motion.div 
            className="orb"
            style={{
              position: 'absolute', bottom: '15%', right: '25%',
              width: '350px', height: '350px',
              background: 'radial-gradient(circle, rgba(59,130,246,0.2) 0%, rgba(0,0,0,0) 70%)',
              filter: 'blur(50px)', zIndex: 0,
              x: mousePosition.x * 2, y: mousePosition.y * 2
            }}
          />

          {/* --- Main Login Card --- */}
          <motion.div
            className="login-card"
            initial={{ y: 30, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.8, type: "spring" }}
            style={{ 
              zIndex: 10,
              textAlign: 'center'
            }}
          >
            {/* Logo Icon */}
            <motion.div 
              initial={{ scale: 0, rotate: -20 }} 
              animate={{ scale: 1, rotate: 0 }} 
              transition={{ delay: 0.2, type: "spring" }}
              style={{ 
                display: 'inline-flex',
                background: 'linear-gradient(135deg, rgba(139, 92, 246, 0.2), rgba(59, 130, 246, 0.2))', 
                padding: '1.2rem', 
                borderRadius: '24px', 
                marginBottom: '1.5rem',
                border: '1px solid rgba(255,255,255,0.1)'
              }}
            >
              <Cpu size={42} color="#a78bfa" strokeWidth={1.5} />
            </motion.div>

            {/* Title */}
            <motion.h1
              className="app-title"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
            >
              NSUT Bot
            </motion.h1>

            {/* Subtitle */}
            <motion.p
              className="app-subtitle"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.4 }}
            >
              Your personal AI academic assistant.
            </motion.p>

            {/* Google Login */}
            <motion.div
              className="google-btn-wrapper"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
            >
              <GoogleLogin
                onSuccess={credentialResponse => {
                  handleLogin(credentialResponse.credential);
                }}
                onError={() => console.log('Login Failed')}
                theme="filled_black"
                shape="pill"
                size="large"
                width="300"
              />
            </motion.div>

            {/* Minimal Footer */}
            <motion.div 
              initial={{ opacity: 0 }} 
              animate={{ opacity: 0.5 }} 
              transition={{ delay: 0.8 }}
              style={{ marginTop: '2.5rem', fontSize: '0.75rem', color: '#94a3b8' }}
            >
              Restricted to @nsut.ac.in domain
            </motion.div>

          </motion.div>
        </div>
      )}
    </>
  )
}

export default App