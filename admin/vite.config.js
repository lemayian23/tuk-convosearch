import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Served under /admin by FastAPI.
//
// app/main.py mounts this build's output at /admin (StaticFiles on
// TUK_ADMIN_DIST_DIR). Without `base`, index.html asks for /assets/index-x.js,
// which is the student SPA's namespace or a 404 — the panel would load a blank
// page. With it, the bundle references /admin/assets/... and works at that path
// on any host, so nothing in the app needs to know where it lives.
//
// Consequence for local dev: the dev server also serves under /admin/, so the
// URL is http://localhost:5173/admin/ (not the bare port). `open` compensates.
//
// Why a single absolute base instead of a relative './': relative asset paths
// work for a SPA without deep links, but they break if StaticFiles ever serves
// this from a path ending without a trailing slash. An explicit '/admin/' is
// the boring, predictable choice.
export default defineConfig({
  base: '/admin/',
  plugins: [react()],
  server: {
    // Land on the right path when you run `npm run dev`.
    open: '/admin/',
  },
  build: {
    // Default 'dist'; pinned here so `deploy/docker-compose.yml`'s bind mount of
    // ./admin/dist -> /srv/app/admin/dist is unambiguous.
    outDir: 'dist',
    emptyOutDir: true,
  },
})
