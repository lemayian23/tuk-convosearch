import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The production build is copied into the API image and served from the same
// origin, so VITE_API_BASE stays empty in deployment. It only needs a value
// when the API runs somewhere else (a phone testing against your laptop).
export default defineConfig({
  plugins: [react()],
  base: '/',
  build: {
    outDir: 'dist',
    // No source maps in the shipped bundle; they are dead weight on a
    // metered connection.
    sourcemap: false,
    modulePreload: { polyfill: false },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.VITE_DEV_API || 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
