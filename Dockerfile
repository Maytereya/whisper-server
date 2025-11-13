FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04
LABEL authors="Rakhmanov"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip libsndfile1 ffmpeg curl && \
    ln -s /usr/bin/python3.11 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /whisper_server
COPY requirements.txt ./
COPY custom_prompt.txt ./

# 👇 ВАЖНО: ставим зависимости именно в тот Python, который будет использоваться
RUN python -m pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /models
ENV MODELS_DIR=/models

COPY whisper_app.py ./

EXPOSE 8000
ENV MODEL=large-v3 DEVICE=cuda COMPUTE_TYPE=float16
CMD ["python", "-m", "uvicorn", "whisper_app:app", "--host", "0.0.0.0", "--port", "8000"]