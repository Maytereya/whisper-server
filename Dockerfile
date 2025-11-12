FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
LABEL authors="Rakhmanov"
# Dockerfile.cuda

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip libsndfile1 ffmpeg && \
    ln -s /usr/bin/python3.11 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /whisper_srv
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /models
ENV MODELS_DIR=/models

COPY whisper_app.py ./

EXPOSE 8000
# Для GPU разумно: float16 или int8_float16
ENV MODEL=large-v3 DEVICE=cuda COMPUTE_TYPE=float16
CMD ["uvicorn", "whisper_app:git app", "--host", "0.0.0.0", "--port", "8000"]