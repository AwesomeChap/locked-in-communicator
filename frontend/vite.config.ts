import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,
    // Proxy /ws to the local Python server during development so the frontend
    // can use the same same-origin URL pattern as production.
    proxy: {
      '/ws': {
        target: 'ws://localhost:8765',
        ws: true,
        rewriteWsOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
});
