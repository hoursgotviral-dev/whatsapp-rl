FROM python:3.11-slim
LABEL maintainer="Dev A"
LABEL description="WhatsApp Sales RL – OpenEnv server"
LABEL version="1.0.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        nginx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt
# Install OpenEnv CLI in an isolated venv to avoid dependency conflicts
# with Gradio runtime dependencies (notably websockets constraints).
RUN python -m venv /opt/openenv-venv \
    && /opt/openenv-venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/openenv-venv/bin/pip install --no-cache-dir openenv-core==0.2.3 \
    && ln -sf /opt/openenv-venv/bin/openenv /usr/local/bin/openenv

COPY . .

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

COPY nginx.conf /etc/nginx/nginx.conf

# Create nginx temp dirs under /tmp so they're writable by any user
RUN mkdir -p /tmp/nginx_client_temp \
             /tmp/nginx_proxy_temp \
             /tmp/nginx_fastcgi_temp \
             /tmp/nginx_uwsgi_temp \
             /tmp/nginx_scgi_temp \
    && sed -i 's/\r$//' start.sh \
    && chmod +x start.sh

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["bash", "start.sh"]
