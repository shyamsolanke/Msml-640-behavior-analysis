FROM python:3.13-slim

# System dependencies required by OpenCV and video processing
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer — only rebuilds when requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Create directories that are expected at runtime
RUN mkdir -p weights data/demo_clip data/mot17 outputs/tracks outputs/metrics outputs/overlays outputs/human_eval

# Expose Streamlit port
EXPOSE 8501

# Streamlit config: headless mode, no CORS, no file watcher
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_PORT=8501

COPY entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
