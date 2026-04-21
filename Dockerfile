# Base: PyTorch 2.1 + CUDA 12.1 + cuDNN 8 (Ampere compatible — RTX A6000)
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

# Prevent interactive prompts during apt
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python dependencies first (cached layer — rebuilds only if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Hugging Face cache inside the container points to a mountable volume
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface

# Tokenizers parallelism warning suppression
ENV TOKENIZERS_PARALLELISM=false
