import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import DashboardPage from './dashboard/page.jsx' 
import Train from './train/page.jsx' 
import './index.css'
import { BrowserRouter, Routes, Route } from 'react-router-dom' // 👈 Import Routes & Route
import { GoogleOAuthProvider } from '@react-oauth/google'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <GoogleOAuthProvider clientId={import.meta.env.VITE_GOOGLE_CLIENT_ID}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<App />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/train" element={<Train />} />
        </Routes>
      </BrowserRouter>
    </GoogleOAuthProvider>
  </React.StrictMode>,
)