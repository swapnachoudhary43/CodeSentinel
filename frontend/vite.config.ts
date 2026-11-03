import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const proxyTarget = process.env.BACKEND_PROXY_TARGET || "http://127.0.0.1:8000"

// Local dev only. Production builds call VITE_API_URL and do not use this proxy.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": proxyTarget,
      "/health": proxyTarget,
    },
  },
})
