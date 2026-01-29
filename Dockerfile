FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel

RUN apt-get update && apt-get install -y --no-install-recommends \
    vim git curl wget ca-certificates \
    libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*
RUN apt-get update
RUN apt-get install -y ffmpeg

COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /workspace
