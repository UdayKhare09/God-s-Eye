import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
  server: {
    proxy: {
      '/register': 'http://127.0.0.1:8000',
      '/recognize': 'http://127.0.0.1:8000',
      '/recognize_video': 'http://127.0.0.1:8000',
      '/set_camera_url': 'http://127.0.0.1:8000',
      '/video_feed': 'http://127.0.0.1:8000',
    }
  },
  build: {
    outDir: '../static',
    emptyOutDir: true,
  }
})
