FROM python:3.11-slim

# yt-dlp merge용 ffmpeg + MediaPipe(OpenCV) 시스템 의존성
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 libgl1 libglib2.0-0 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# MotionBERT 비교 분기가 필요하면 주석 해제 (CPU 빌드, +~200MB)
# RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY app ./app
COPY main.py ./
COPY frontend ./frontend
RUN mkdir -p tmp

ENV HOST=0.0.0.0 PORT=8000 TMP_DIR=/srv/tmp PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -f http://127.0.0.1:8000/health || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
