#!/bin/bash
set -euo pipefail

# Normalize API key env vars so gradio demo can use hosted secrets
if [ -z "${HF_TOKEN:-}" ] && [ -n "${HUGGINGFACEHUB_API_TOKEN:-}" ]; then
    export HF_TOKEN="${HUGGINGFACEHUB_API_TOKEN}"
fi
if [ -z "${HF_TOKEN:-}" ] && [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
fi
if [ -z "${API_KEY:-}" ] && [ -n "${OPENAI_API_KEY:-}" ]; then
    export API_KEY="${OPENAI_API_KEY}"
fi

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
nginx -g "daemon off;" &
NGINX_PID=$!

# Keep container healthy: if any child process dies, stop the rest and exit.
wait -n -p EXITED_PID "${UVICORN_PID}" "${GRADIO_PID}" "${NGINX_PID}" || true
echo "[start.sh] Process ${EXITED_PID} exited. Shutting down remaining services..."

kill "${UVICORN_PID}" "${GRADIO_PID}" "${NGINX_PID}" 2>/dev/null || true
wait "${UVICORN_PID}" "${GRADIO_PID}" "${NGINX_PID}" 2>/dev/null || true
exit 1
