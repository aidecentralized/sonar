import { defineConfig } from 'vite';
import fs from 'fs';
import path from 'path';

// vite.config.js
export default defineConfig({
  server: {
    host: '0.0.0.0', // Bind to all network interfaces
    port: process.env.PORT || 10000, // Use Render's provided port
    strictPort: true, // Ensures Vite fails if the port is unavailable
    allowedHosts: ['.onrender.com'], // Allow all Render subdomains
    https: process.env.Render? false : {
      key: fs.readFileSync(path.resolve(__dirname, 'certs/localhost+3-key.pem')),
      cert: fs.readFileSync(path.resolve(__dirname, 'certs/localhost+3.pem')),  
    }
  }
});