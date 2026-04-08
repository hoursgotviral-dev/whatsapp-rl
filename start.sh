fixed start.sh: #!/bin/bash
set -e

echo "[start.sh] Starting Gradio on port 7861..."
python ui/gradio_demo.py &
GRADIO_PID=$!

echo "[start.sh] Starting FastAPI on port 8000..."
uvicorn server:app --host 0.0.0.0 --port 8000 &
UVICORN_PID=$!

echo "[start.sh] Waiting for services to initialise..."
sleep 5

# Verify both processes are still running
if ! kill -0 $GRADIO_PID 2>/dev/null; then
    echo "[start.sh] ERROR: Gradio failed to start"
    exit 1
fi
if ! kill -0 $UVICORN_PID 2>/dev/null; then
    echo "[start.sh] ERROR: FastAPI failed to start"
    exit 1
fi

echo "[start.sh] All services up — starting nginx on port 7860"
nginx -g "daemon off;"