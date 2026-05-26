FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive TZ=UTC PYTHONUNBUFFERED=1

# Sistema base
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-dev git wget curl unzip ffmpeg \
    nodejs npm build-essential cmake ninja-build \
    libgl1-mesa-glx libglib2.0-0 \
    libboost-program-options-dev libboost-filesystem-dev \
    libboost-graph-dev libboost-system-dev libeigen3-dev \
    libflann-dev libfreeimage-dev libmetis-dev libgoogle-glog-dev \
    libgtest-dev libsqlite3-dev libglew-dev qtbase5-dev \
    libqt5opengl5-dev libcgal-dev libceres-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && pip install --upgrade pip setuptools wheel

# COLMAP 3.9.1 desde fuente (garantiza compatibilidad CUDA 12.1)
RUN git clone --branch 3.9.1 --depth 1 https://github.com/colmap/colmap.git /tmp/colmap && \
    cd /tmp/colmap && mkdir build && cd build && \
    cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90" \
    -DCMAKE_INSTALL_PREFIX=/usr/local && \
    ninja -j$(nproc) && ninja install && rm -rf /tmp/colmap

# PyTorch + CUDA 12.1
RUN pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121

# Dependencias Python
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# gsplat (compila kernels CUDA)
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0" MAX_JOBS=4
RUN pip install --no-cache-dir gsplat==1.4.0 || pip install --no-cache-dir gsplat

# Trainer de gsplat (simple_trainer.py)
RUN git clone --depth 1 https://github.com/nerfstudio-project/gsplat.git /opt/gsplat-repo && \
    pip install --no-cache-dir -r /opt/gsplat-repo/examples/requirements.txt 2>/dev/null || true

# splat-transform para collision mesh
RUN npm install -g @playcanvas/splat-transform 2>/dev/null || true

# Pre-descargar modelos AI (evita descargas en cada job)
RUN python -c "from transformers import pipeline; \
    p=pipeline('depth-estimation',model='depth-anything/Depth-Anything-V2-Small-hf'); \
    print('Depth Anything V2 OK')" 2>/dev/null || true

RUN python -c "from transformers import Mask2FormerImageProcessor,Mask2FormerForUniversalSegmentation; \
    Mask2FormerImageProcessor.from_pretrained('facebook/mask2former-swin-base-ade-semantic'); \
    Mask2FormerForUniversalSegmentation.from_pretrained('facebook/mask2former-swin-base-ade-semantic'); \
    print('Mask2Former OK')" 2>/dev/null || true

COPY handler.py .
RUN mkdir -p /workspace/logs /workspace/jobs
CMD ["python","-u","handler.py"]
