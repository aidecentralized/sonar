// vite.config.js
export default {
  server: {
    host: '0.0.0.0', // Bind to all network interfaces
    port: process.env.PORT || 10000, // Use Render's provided port
    strictPort: true // Ensures Vite fails if the port is unavailable
    allowedHosts: ['.onrender.com'], // Allow all Render subdomains
  }
}