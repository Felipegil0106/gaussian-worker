# ══════════════════════════════════════════════════════════════
# Imagen Gaussian Scanner — Malla (COLMAP + OpenMVS, todo precompilado)
# ══════════════════════════════════════════════════════════════
# Base: la MISMA que ya usamos y funciona (torch 2.1.1 + CUDA 12.1 + Ubuntu 22.04).
# Sobre ella agregamos COLMAP (apt) y OpenMVS (compilado con la receta oficial).
#
# Construir:  docker build -t TU_USUARIO/gaussian-mesh:v1 .
# Subir:      docker push TU_USUARIO/gaussian-mesh:v1
# ══════════════════════════════════════════════════════════════
FROM runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ── Paso 1: sistema base + COLMAP + dependencias de OpenMVS ──
# (COLMAP de apt = el que ya usamos y sirve para la parte sparse)
RUN apt-get update -yq && apt-get install -yq \
    build-essential git cmake wget ffmpeg pkg-config \
    colmap xvfb \
    libpng-dev libjpeg-dev libtiff-dev \
    libglu1-mesa-dev libglew-dev libglfw3-dev \
    libboost-iostreams-dev libboost-program-options-dev \
    libboost-system-dev libboost-serialization-dev libboost-thread-dev \
    libopencv-dev \
    libgmp-dev libmpfr-dev zlib1g-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Paso 2: Eigen 3.4 (versión EXACTA que pide OpenMVS) ──
RUN git clone https://gitlab.com/libeigen/eigen --branch 3.4 /tmp/eigen && \
    mkdir /tmp/eigen_build && cd /tmp/eigen_build && \
    cmake . /tmp/eigen -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda/ && \
    make && make install && \
    cd / && rm -rf /tmp/eigen_build /tmp/eigen

# ── Paso 3: CGAL v6.0.1 (versión EXACTA de la receta oficial) ──
RUN git clone https://github.com/cgal/cgal --branch=v6.0.1 /tmp/cgal && \
    mkdir /tmp/cgal_build && cd /tmp/cgal_build && \
    cmake . /tmp/cgal && \
    make && make install && \
    cd / && rm -rf /tmp/cgal_build /tmp/cgal

# ── Paso 4: VCGLib (no se compila, solo se clona y se referencia) ──
RUN git clone https://github.com/cdcseacave/VCG.git /opt/vcglib

# ── Paso 4b: nanoflann (dependencia nueva de OpenMVS master) ──
# Es header-only, compila en segundos. Instala el nanoflannConfig.cmake
# que OpenMVS busca con FIND_PACKAGE(nanoflann REQUIRED).
RUN git clone https://github.com/jlblancoc/nanoflann.git --branch v1.5.5 /tmp/nanoflann && \
    mkdir /tmp/nanoflann_build && cd /tmp/nanoflann_build && \
    cmake . /tmp/nanoflann \
        -DNANOFLANN_BUILD_EXAMPLES=OFF \
        -DNANOFLANN_BUILD_TESTS=OFF && \
    make install && \
    cd / && rm -rf /tmp/nanoflann_build /tmp/nanoflann

# ── Paso 5: OpenMVS (compilar con CUDA, rama master estable) ──
RUN git clone https://github.com/cdcseacave/openMVS.git --branch master /tmp/openMVS && \
    sed -i 's/pkg_check_modules(${PREFIX} REQUIRED IMPORTED_TARGET ${MODULE_NAME})/pkg_check_modules(${PREFIX} IMPORTED_TARGET ${MODULE_NAME})/' /tmp/openMVS/libs/IO/CMakeLists.txt && \
    mkdir /tmp/openMVS_build && cd /tmp/openMVS_build && \
    cmake . /tmp/openMVS \
        -DCMAKE_BUILD_TYPE=Release \
        -DVCG_ROOT=/opt/vcglib \
        -DOpenMVS_USE_CUDA=ON \
        -DOpenMVS_BUILD_VIEWER=OFF \
        -DOpenMVS_USE_BREAKPAD=OFF \
        -DOpenMVS_ENABLE_TESTS=OFF \
        -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs/ \
        -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda/ \
        -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89" \
        -DEIGEN3_INCLUDE_DIR=/usr/local/include/eigen3 && \
    make -j$(nproc) && \
    make install && \
    cd / && rm -rf /tmp/openMVS_build /tmp/openMVS

# ── Paso 6: librerías Python que el worker necesita ──
# (para descargar/subir, procesar imágenes y convertir la malla a .glb)
RUN pip install --no-cache-dir \
    boto3 requests tqdm pillow "numpy<2" \
    opencv-python-headless trimesh

# Los binarios de OpenMVS quedan en /usr/local/bin/OpenMVS
ENV PATH=/usr/local/bin/OpenMVS:$PATH

# Verificación: que los ejecutables existan (no debe hacer fallar el build)
RUN echo "=== Verificando OpenMVS ===" && \
    ls -la /usr/local/bin/OpenMVS/ && \
    echo "=== OpenMVS instalado OK ==="

WORKDIR /workspace
