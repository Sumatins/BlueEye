# BlueEye - CPU Docker image.
# For GPU inference, change the base image to an NVIDIA CUDA image and run
# with the NVIDIA container toolkit (see README -> "GPU Support"), e.g.:
#   FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BLUEEYE_LOG_LEVEL=INFO

WORKDIR /blueeye

# System libraries required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY src ./src
COPY assets ./assets
COPY README.md .

# Model weights, uploads and results live in mounted volumes so they
# survive container restarts (download weights at runtime).
RUN mkdir -p models data/input outputs/images outputs/videos outputs/reports

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import sys, urllib.request; \
r = urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=5); \
sys.exit(0 if r.status == 200 else 1)"

# Download weights inside the container with:
#   docker compose run --rm blueeye python scripts/download_models.py
CMD ["streamlit", "run", "app/ui/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
