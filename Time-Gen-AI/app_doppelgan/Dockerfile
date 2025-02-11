# Dockerfile
FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

WORKDIR /app

# Install dependencies
RUN apt-get update && \
    apt-get install -y python3.10 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

COPY app_rtsgan/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .
RUN mv app/app_timgan.py app/app.py

# Environment variables
ENV PYTHONPATH=/app:/app/app
ENV HF_HOME=/app/huggingface_cache
ENV MODEL_TYPE=doppelgan

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]