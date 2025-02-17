# Use CUDA 11.8 base image with cuDNN 8
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive
# Install system dependencies and Python 3.9
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    wget \
    ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.9
RUN wget https://bootstrap.pypa.io/get-pip.py \
    && python3.9 get-pip.py \
    && rm get-pip.py

# Set up Python 3.9 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

# Install remaining system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python packages with correct pip references
RUN python3.9 -m pip install --no-cache-dir --upgrade pip && \
    python3.9 -m pip install --no-cache-dir \
    git+https://github.com/gretelai/gretel-synthetics.git@v0.22.16 \
    torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117 \
    "numpy>=1.17.2" \
    tensorflow==2.10 \
    "tqdm>=4.36.1" \
    "argparse>=1.1" \
    "pandas>=0.25.1" \
    "scikit-learn>=0.21.3" \
    "matplotlib>=3.1.1" \
    protobuf==3.19.3 \
    fastNLP==0.5.5 \
    einops \
    imageio \
    tf-slim \
    tensorflow-addons \
    opencv-python

# Verification commands
RUN python3 -c "import pip; print(f'Pip version: {pip.__version__}')" && \
    python3 -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}')" && \
    python3 -c "import tensorflow as tf; print(f'TensorFlow GPUs: {len(tf.config.list_physical_devices(\"GPU\"))}')"

CMD ["python3"]