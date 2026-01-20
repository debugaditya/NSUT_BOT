import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // 1. Fix white screen on subdirectories (e.g., GitHub Pages)
  base: '/', 
  server: {
    // 2. Fix Google Login 'postMessage' / COOP errors
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin-allow-popups',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  build: {
    // 3. Ensure the output directory matches your deployment (standard is dist)
    outDir: 'dist',
  }
})