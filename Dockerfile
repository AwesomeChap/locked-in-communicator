# ── Stage 1: build the React frontend ────────────────────────────────────────
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm install

COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python runtime ───────────────────────────────────────────────────
FROM python:3.11-slim AS runtime
WORKDIR /app

# Install Python deps
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install the backend package (makes `lockedin-verification-server` available)
COPY backend/ ./backend/
RUN pip install --no-cache-dir -e ./backend

# Copy the compiled React app so the server can serve it
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# ── Runtime config ────────────────────────────────────────────────────────────
ENV WS_HOST=0.0.0.0
ENV STATIC_DIR=frontend/dist
# 10000 = Render's conventional Docker port (also overridable for HF Spaces etc.)
ENV PORT=10000

EXPOSE 10000

CMD ["lockedin-verification-server"]
