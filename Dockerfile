FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive TZ=UTC PYTHONUNBUFFERED=1
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV FORCE_CUDA="1"
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
ENV MAX_JOBS=4

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

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    pip install --upgrade pip setuptools wheel

# COLMAP 3.9.1 desde fuente
RUN git clone --branch 3.9.1 --depth 1 https://github.com/colmap/colmap.git /tmp/colmap && \
    cd /tmp/colmap && mkdir build && cd build && \
    cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90" \
    -DCMAKE_INSTALL_PREFIX=/usr/local && \
    ninja -j$(nproc) && ninja install && rm -rf /tmp/colmap

# PyTorch 2.1.2 + CUDA 12.1
RUN pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121

# Ninja para acelerar compilación
RUN pip install --no-cache-dir ninja

# Dependencias Python
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ════════════════════════════════════════════════════════════════
# FIX CRÍTICO: gsplat librería Y trainer DEBEN ser la MISMA versión
# Antes: pip instalaba 1.4.0 pero git clone traía 'main' (desajuste).
# Ahora: ambos fijados al tag v1.4.0 exactamente.
# ════════════════════════════════════════════════════════════════

# Clonar gsplat v1.4.0 EXACTO (tag) y instalarlo DESDE esa fuente
RUN git clone --branch v1.4.0 --depth 1 \
    https://github.com/nerfstudio-project/gsplat.git /opt/gsplat-repo && \
    cd /opt/gsplat-repo && \
    pip install --no-cache-dir . && \
    pip install --no-cache-dir -r examples/requirements.txt

# splat-transform para collision mesh
RUN npm install -g @playcanvas/splat-transform 2>/dev/null || true

# Pre-descargar modelos AI
RUN python -c "from transformers import pipeline; \
    p=pipeline('depth-estimation',model='depth-anything/Depth-Anything-V2-Small-hf'); \
    print('Depth Anything V2 OK')" 2>/dev/null || true

RUN python -c "from transformers import Mask2FormerImageProcessor,Mask2FormerForUniversalSegmentation; \
    Mask2FormerImageProcessor.from_pretrained('facebook/mask2former-swin-base-ade-semantic'); \
    Mask2FormerForUniversalSegmentation.from_pretrained('facebook/mask2former-swin-base-ade-semantic'); \
    print('Mask2Former OK')" 2>/dev/null || true

COPY handler.py .
RUN mkdir -p /workspace/logs /workspace/jobs

CMD ["python", "-u", "handler.py"]
