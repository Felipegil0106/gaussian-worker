FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# 1. Asegurar variables críticas
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV FORCE_CUDA="1"
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"

# 2. Instalación de herramientas base (Git es prioridad)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential cmake ninja-build \
    python3.10 python3-pip python3.10-dev \
    ffmpeg libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 3. Verificar que git existe
RUN git --version

# 4. Preparar Python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    pip install --upgrade pip setuptools wheel

# 5. Instalar PyTorch
RUN pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121

# 6. Instalar gsplat (Compilación robusta)
# Instalamos ninja primero para acelerar la compilación
RUN pip install --no-cache-dir ninja
RUN pip install --no-cache-dir git+https://github.com/nerfstudio-project/gsplat.git

# 7. Dependencias restantes
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 8. Trainer de gsplat
RUN git clone --depth 1 https://github.com/nerfstudio-project/gsplat.git /opt/gsplat-repo

# 9. Copiar el resto
COPY handler.py .
RUN mkdir -p /workspace/logs /workspace/jobs
CMD ["python","-u","handler.py"]
