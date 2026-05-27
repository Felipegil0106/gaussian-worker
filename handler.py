"""
Worker GPU para Gaussian Splatting — Pipeline de 8 etapas
Recibe ZIP de fotos, devuelve .ply + .glb (collision mesh)
"""

import os, sys, json, shutil, subprocess, zipfile, tempfile, traceback, time, base64
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import requests as req
import runpod

# ── Configuración ──────────────────────────────────────────────

TIMEOUTS = {
    "download": 600, "colmap_feature": 600, "colmap_match": 900,
    "colmap_mapper": 1800, "colmap_undistort": 300,
    "gsplat": 2700, "collision": 600, "upload": 600,
}

ITERS = {"fast": 7000, "balanced": 30000, "quality": 50000}

BLUR_THRESHOLD_ABSOLUTE = 30.0
BLUR_PERCENTILE_FALLBACK = 25
MIN_VALID_RATIO = 0.5
MIN_IMGS, MAX_IMGS = 20, 1000

# ── Logging ────────────────────────────────────────────────────

LOG_PATH = "/workspace/jobs/job.log"
CMD_PATH = "/workspace/jobs/last_cmd.txt"

class Log:
    def __init__(self):
        self.t0 = time.time()
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        open(LOG_PATH, "w").write(f"=== {datetime.now(timezone.utc).isoformat()} ===\n")

    def __call__(self, msg, lv="INFO"):
        line = f"[{lv}][+{time.time()-self.t0:.1f}s] {msg}"
        print(line, flush=True)
        try: open(LOG_PATH, "a").write(line+"\n")
        except: pass

    def tail(self, n=60):
        try: return "\n".join(open(LOG_PATH).readlines()[-n:])
        except: return ""

log = Log()

def save_cmd(cmd, err=None):
    try:
        with open(CMD_PATH, "w") as f:
            f.write(f"CMD: {' '.join(str(c) for c in cmd) if isinstance(cmd,list) else cmd}\n")
            if err: f.write(f"ERR: {err}\n")
    except: pass

def get_cmd():
    try: return open(CMD_PATH).read()
    except: return ""

def run(cmd, timeout, name=""):
    save_cmd(cmd)
    log(f"[{name}] {' '.join(str(c) for c in cmd[:6])}")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        raise Timeout(f"Timeout {name} ({timeout}s)")
    if r.returncode != 0:
        err = "\n".join(r.stderr.split("\n")[-15:])
        save_cmd(cmd, err)
        raise RuntimeError(f"[{name}] rc={r.returncode}\n{err}")
    return r.stdout

def download(url, dest, timeout=600):
    log("Descargando...")
    r = req.get(url, stream=True, timeout=(30, timeout))
    r.raise_for_status()
    dl = 0
    with open(dest, "wb") as f:
        for chunk in r.iter_content(1024*1024):
            f.write(chunk)
            dl += len(chunk)
    log(f"Descargado: {dl//(1024*1024)} MB")

def upload(path, url, timeout=600):
    log(f"Subiendo {os.path.basename(path)} ({os.path.getsize(path)//(1024*1024)} MB)...")
    with open(path, "rb") as f:
        r = req.put(url, data=f, timeout=timeout, headers={"Content-Type": "application/octet-stream"})
        r.raise_for_status()
    log("Upload OK")

# ── Etapa 1: Extracción ─────────────────────────────

def extract_frames(raw_dir, frames_dir):
    log("━━━ ETAPA 1: Extracción de frames ━━━")
    files = os.listdir(raw_dir)
    videos = [f for f in files if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))]
    images = sorted([f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    if videos:
        log(f"Video: {videos[0]}")
        run(["ffmpeg", "-i", os.path.join(raw_dir, videos[0]),
             "-vf", "mpdecimate=hi=64*12:lo=64*5:frac=0.33,fps=2",
             "-qscale:v", "2", "-vsync", "vfr",
             os.path.join(frames_dir, "frame_%05d.jpg")], 600, "ffmpeg")
    elif images:
        log(f"Fotos: {len(images)}")
        for i, img in enumerate(images):
            shutil.copy(os.path.join(raw_dir, img), os.path.join(frames_dir, f"frame_{i:05d}.jpg"))
    else:
        raise RuntimeError("No hay videos ni imágenes en el ZIP")
    return len([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])

# ── Etapa 2: Blur ───────────────────────────

def filter_blur(frames_dir):
    log("━━━ ETAPA 2: Filtro de blur ━━━")
    import cv2
    frames = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    if not frames: return 0
    variances = []
    for f in frames:
        img = cv2.imread(os.path.join(frames_dir, f), cv2.IMREAD_GRAYSCALE)
        if img is None: variances.append((f, 0.0)); continue
        variances.append((f, cv2.Laplacian(img, cv2.CV_64F).var()))
    
    vars_only = [v for _, v in variances]
    above_absolute = sum(1 for _, v in variances if v >= BLUR_THRESHOLD_ABSOLUTE)
    
    if (above_absolute / len(variances)) >= MIN_VALID_RATIO:
        to_remove = [f for f, v in variances if v < BLUR_THRESHOLD_ABSOLUTE]
    else:
        sorted_vars = sorted(vars_only)
        adaptive_threshold = sorted_vars[max(1, len(sorted_vars) * BLUR_PERCENTILE_FALLBACK // 100)]
        to_remove = [f for f, v in variances if v < adaptive_threshold]
    
    if len(to_remove) >= len(frames): to_remove = []
    for f in to_remove: os.remove(os.path.join(frames_dir, f))
    return len(frames) - len(to_remove)

# ── Etapas 3, 4, 5 y 6 ────────────────────────────────

def gen_depth(frames_dir, depth_dir):
    log("━━━ ETAPA 3: Depth Anything V2 ━━━")
    # ... (Tu código existente de Depth) ...

def gen_masks(frames_dir, masks_dir):
    log("━━━ ETAPA 4: Mask2Former ━━━")
    # ... (Tu código existente de Mask2Former) ...

def run_colmap(images_dir, output_dir):
    log("━━━ ETAPA 5: COLMAP ━━━")
    # ... (Tu código existente de COLMAP) ...

def run_gsplat(data_dir, result_dir, iters):
    log(f"━━━ ETAPA 6: gsplat ({iters} iter) ━━━")
    # IMPORTANTE: Eliminamos el pip install. El Dockerfile ya lo incluye.
    
    trainer = "/opt/gsplat-repo/examples/simple_trainer.py"
    if not os.path.exists(trainer):
        # Fallback si por alguna razón no está en /opt
        trainer = "/workspace/gsplat-repo/examples/simple_trainer.py"

    run(["python", trainer, "default",
         "--data_dir", data_dir, "--data_factor", "1",
         "--result_dir", result_dir,
         "--max_steps", str(iters),
         "--save_steps", str(iters),
         "--eval_steps", str(iters+1),
         "--disable_viewer"],
        TIMEOUTS["gsplat"], "gsplat")
    log("gsplat OK")

# ── Handler Principal (Idéntico a tu anterior versión) ──
# (Asegúrate de copiar el resto del archivo: Etapas 7, 8 y handler)
