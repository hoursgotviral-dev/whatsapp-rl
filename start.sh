#!/bin/bash
set -euo pipefail

echo "[start.sh] Starting FastAPI on port 8000..."
uvicorn server.app:app --host 0.0.0.0 --port 8000 &
UVICORN_PID=$!

echo "[start.sh] Starting Gradio on port 7861..."
python app/launch_gradio.py &
GRADIO_PID=$!

echo "[start.sh] Waiting for services to initialise..."
sleep 5

if ! kill -0 "${UVICORN_PID}" 2>/dev/null; then
    echo "[start.sh] ERROR: FastAPI failed to start"
    exit 1
fi
if ! kill -0 "${GRADIO_PID}" 2>/dev/null; then
    echo "[start.sh] ERROR: Gradio failed to start"
    exit 1
fi

echo "[start.sh] Starting nginx on port 7860..."
nginx -g "daemon off;"
