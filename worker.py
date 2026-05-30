#!/usr/bin/env python3
"""
Gaussian Worker v3 — Pod-Standalone Edition
═══════════════════════════════════════════════════════════════════
ES tu handler.py de siempre (gsplat 1.4.0, 8 etapas COMPLETAS) pero
re-empaquetado para ejecutarse dentro de un POD RunPod en lugar de
Serverless.

CAMBIOS vs handler.py:
  - Lee parámetros desde env vars (no de un dict 'job')
  - Manda callbacks HMAC al backend (progress, completed, error)
  - Heartbeat cada 30s
  - En error, manda el LOG COMPLETO al backend (descargable después)
  - Es un script con main(), no una función handler() de RunPod

NO CAMBIA (sigue idéntico a tu handler.py):
  - Las 8 etapas del pipeline
  - El filtro adaptativo de blur
  - gsplat v1.4.0 con simple_trainer del MISMO tag (sin bug color_correct)
  - Depth Anything V2 + Mask2Former
  - COLMAP 3.9.x desde apt
"""

import os, sys, json, time, hmac, hashlib, threading, traceback
import shutil, subprocess, zipfile, tempfile
from pathlib import Path
from datetime import datetime, timezone
from collections import deque

import numpy as np
import requests as req

# ══════════════════════════════════════════════════════════════
# CONFIGURACIÓN — todo desde env vars (el backend las setea)
# ══════════════════════════════════════════════════════════════

TOUR_ID        = os.environ["TOUR_ID"]                 # id único del job
INPUT_URL      = os.environ["INPUT_URL"]               # URL presignada del ZIP en R2
UPLOAD_URL_PLY = os.environ["UPLOAD_URL_PLY"]          # URL presignada para subir .ply
UPLOAD_URL_GLB = os.environ.get("UPLOAD_URL_GLB", "")  # opcional para .glb
CALLBACK_URL   = os.environ["CALLBACK_URL"]            # https://railway.../api/internal/callback/{TOUR_ID}
CALLBACK_SECRET = os.environ["CALLBACK_SECRET"]        # firma HMAC
QUALITY        = os.environ.get("QUALITY", "fast")     # fast | balanced | quality

TIMEOUTS = {
    "download":600, "colmap_feature":600, "colmap_match":900,
    "colmap_mapper":1800, "colmap_undistort":300,
    "gsplat":2700, "collision":600, "upload":600,
}
ITERS = {"fast":7000, "balanced":30000, "quality":50000}

BLUR_THRESHOLD_ABSOLUTE = 30.0
BLUR_PERCENTILE_FALLBACK = 25
MIN_VALID_RATIO = 0.5
MIN_IMGS, MAX_IMGS = 20, 1000

WORK = Path("/workspace/job")
WORK.mkdir(parents=True, exist_ok=True)
INPUT_ZIP    = WORK / "input.zip"
RAW_DIR      = WORK / "raw"
FRAMES_DIR   = WORK / "frames"
DEPTH_DIR    = WORK / "depth"
MASKS_DIR    = WORK / "masks"
COLMAP_DIR   = WORK / "colmap"
RESULT_DIR   = WORK / "result"
for d in (RAW_DIR, FRAMES_DIR, COLMAP_DIR, RESULT_DIR):
    d.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════
# LOGGING (buffer para mandar al backend si hay error)
# ══════════════════════════════════════════════════════════════

_LOG = deque(maxlen=500)
_t0 = time.time()
_current_progress = 0.0
_current_message = "Iniciando..."
_keep_heartbeat = True

def log(msg, lv="INFO"):
    line = f"[{lv}][+{time.time()-_t0:.1f}s] {msg}"
    _LOG.append(line)
    print(line, flush=True)

def full_log():
    return "\n".join(_LOG)

# ══════════════════════════════════════════════════════════════
# CALLBACK con firma HMAC (igual que Vessel)
# ══════════════════════════════════════════════════════════════

def _clean_url(url):
    if not url: return ""
    url = url.strip()
    for _ in range(5):
        changed = False
        for bad in ("http:https://", "https:https://", "http:http://", "https:http://"):
            if url.startswith(bad):
                url = url[len(bad)-len("https://"):]
                changed = True; break
        if not changed: break
    if not url.startswith(("http://","https://")): url = "https://" + url
    return url.rstrip("/")

CALLBACK_URL = _clean_url(CALLBACK_URL)
log(f"CALLBACK_URL: {CALLBACK_URL}")

def callback(payload):
    body = json.dumps(payload).encode()
    sig = hmac.new(CALLBACK_SECRET.encode(), body, hashlib.sha256).hexdigest()
    try:
        r = req.post(CALLBACK_URL, data=body, timeout=15,
                     headers={"Content-Type":"application/json", "X-Signature":sig})
        if r.status_code != 200:
            log(f"callback non-200: {r.status_code} {r.text[:200]}", "WARN")
        return r.status_code == 200
    except Exception as e:
        log(f"callback failed: {e}", "WARN")
        return False

def report(progress, message):
    global _current_progress, _current_message
    _current_progress = progress
    _current_message = message
    log(f"[{progress*100:.0f}%] {message}")
    callback({"type":"progress", "progress":progress, "message":message})

def heartbeat_loop():
    while _keep_heartbeat:
        try:
            callback({"type":"progress", "progress":_current_progress, "message":_current_message})
        except: pass
        time.sleep(30)

# ══════════════════════════════════════════════════════════════
# UTILIDADES
# ══════════════════════════════════════════════════════════════

class Timeout(Exception): pass

def _has_xvfb():
    """Detecta si xvfb-run está disponible (pantalla virtual para COLMAP)."""
    return shutil.which("xvfb-run") is not None

def run(cmd, timeout, name="", use_xvfb=False):
    # FIX v3.8: COLMAP (Qt) necesita un display aunque corra headless.
    # Sin pantalla, aborta con 'QGuiApplicationPrivate::createPlatformIntegration' rc=-6.
    # xvfb-run crea una pantalla virtual en RAM y resuelve el crash.
    if use_xvfb and _has_xvfb():
        cmd = ["xvfb-run", "-a", "-s", "-screen 0 1280x1024x24"] + list(cmd)
    log(f"[{name}] " + " ".join(str(c) for c in cmd[:8]))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        raise Timeout(f"Timeout {name} ({timeout}s)")
    if r.returncode != 0:
        err = "\n".join(r.stderr.split("\n")[-15:])
        raise RuntimeError(f"[{name}] rc={r.returncode}\n{err}")
    return r.stdout

def download(url, dest, timeout=600):
    log("Descargando ZIP...")
    r = req.get(url, stream=True, timeout=(30, timeout))
    r.raise_for_status()
    dl = 0
    with open(dest, "wb") as f:
        for chunk in r.iter_content(1024*1024):
            f.write(chunk); dl += len(chunk)
    log(f"Descargado: {dl//(1024*1024)} MB")

def upload(path, url, timeout=600):
    log(f"Subiendo {os.path.basename(path)} ({os.path.getsize(path)//(1024*1024)} MB)...")
    with open(path, "rb") as f:
        r = req.put(url, data=f, timeout=timeout,
                    headers={"Content-Type":"application/octet-stream"})
        r.raise_for_status()
    log("Upload OK")

# ══════════════════════════════════════════════════════════════
# ETAPA 1: EXTRACCIÓN DE FRAMES
# ══════════════════════════════════════════════════════════════

def extract_frames():
    log("━━━ ETAPA 1: Extracción de frames ━━━")
    files = os.listdir(RAW_DIR)
    videos = [f for f in files if f.lower().endswith((".mp4",".mov",".avi",".mkv"))]
    images = sorted([f for f in files if f.lower().endswith((".jpg",".jpeg",".png"))])

    if videos:
        log(f"Video: {videos[0]}")
        run(["ffmpeg","-i",str(RAW_DIR / videos[0]),
             "-vf","mpdecimate=hi=64*12:lo=64*5:frac=0.33,fps=2",
             "-qscale:v","2","-vsync","vfr",
             str(FRAMES_DIR / "frame_%05d.jpg")], 600, "ffmpeg")
    elif images:
        log(f"Fotos: {len(images)}")
        for i, img in enumerate(images):
            shutil.copy(str(RAW_DIR / img), str(FRAMES_DIR / f"frame_{i:05d}.jpg"))
    else:
        raise RuntimeError("No hay videos ni imágenes en el ZIP")

    count = len([f for f in os.listdir(FRAMES_DIR) if f.endswith(".jpg")])
    log(f"Frames: {count}")
    return count

# ══════════════════════════════════════════════════════════════
# ETAPA 2: FILTRO BLUR ADAPTATIVO
# ══════════════════════════════════════════════════════════════

def filter_blur():
    log("━━━ ETAPA 2: Filtro de blur ADAPTATIVO ━━━")
    import cv2
    frames = sorted([f for f in os.listdir(FRAMES_DIR) if f.endswith(".jpg")])
    if not frames: return 0

    variances = []
    for f in frames:
        img = cv2.imread(str(FRAMES_DIR / f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            variances.append((f, 0.0)); continue
        v = cv2.Laplacian(img, cv2.CV_64F).var()
        variances.append((f, v))

    vals = [v for _, v in variances]
    log(f"Varianzas: min={min(vals):.1f} max={max(vals):.1f} "
        f"media={sum(vals)/len(vals):.1f} mediana={sorted(vals)[len(vals)//2]:.1f}")

    above = sum(1 for _, v in variances if v >= BLUR_THRESHOLD_ABSOLUTE)
    ratio = above / len(variances)
    log(f"Sobre umbral ({BLUR_THRESHOLD_ABSOLUTE}): {above}/{len(variances)} ({ratio*100:.0f}%)")

    if ratio >= MIN_VALID_RATIO:
        log(f"Modo NORMAL")
        to_remove = [f for f, v in variances if v < BLUR_THRESHOLD_ABSOLUTE]
    else:
        sv = sorted(vals)
        adapt = sv[max(1, len(sv) * BLUR_PERCENTILE_FALLBACK // 100)]
        log(f"Modo PERMISIVO (umbral adaptativo {adapt:.1f})")
        to_remove = [f for f, v in variances if v < adapt]

    if len(to_remove) >= len(frames):
        to_remove = []
        log("WARN: hubiera eliminado todo. Conservando todos.", "WARN")

    if len(frames) - len(to_remove) < MIN_IMGS:
        best = sorted(variances, key=lambda x: x[1], reverse=True)
        keep = set(f for f, _ in best[:MIN_IMGS])
        to_remove = [f for f in frames if f not in keep]
        log(f"WARN: conservando los {MIN_IMGS} mejores", "WARN")

    for f in to_remove:
        os.remove(str(FRAMES_DIR / f))
    kept = len(frames) - len(to_remove)
    log(f"Nítidas: {kept}, Borrosas eliminadas: {len(to_remove)}")
    return kept

# ══════════════════════════════════════════════════════════════
# ETAPA 3: DEPTH ANYTHING V2
# ══════════════════════════════════════════════════════════════

def gen_depth():
    log("━━━ ETAPA 3: Depth Anything V2 ━━━")
    DEPTH_DIR.mkdir(exist_ok=True)
    try:
        import torch
        from PIL import Image
        from transformers import pipeline
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"Device: {device}")
        pipe = pipeline("depth-estimation",
                       model="depth-anything/Depth-Anything-V2-Small-hf",
                       device=device)
        frames = sorted([f for f in os.listdir(FRAMES_DIR) if f.endswith(".jpg")])
        for i, f in enumerate(frames):
            img = Image.open(str(FRAMES_DIR / f)).convert("RGB")
            d = pipe(img)["depth"]
            arr = np.array(d)
            norm = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 65535).astype(np.uint16)
            Image.fromarray(norm).save(str(DEPTH_DIR / f.replace(".jpg","_depth.png")))
            if (i+1) % 25 == 0:
                log(f"  Depth {i+1}/{len(frames)}")
        log(f"Depth maps: {len(frames)}")
    except Exception as e:
        log(f"WARN Depth falló: {e}", "WARN")

# ══════════════════════════════════════════════════════════════
# ETAPA 4: MASK2FORMER
# ══════════════════════════════════════════════════════════════

def gen_masks():
    log("━━━ ETAPA 4: Mask2Former ━━━")
    MASKS_DIR.mkdir(exist_ok=True)
    try:
        import torch
        from PIL import Image
        from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
        device = "cuda" if torch.cuda.is_available() else "cpu"
        proc = Mask2FormerImageProcessor.from_pretrained(
            "facebook/mask2former-swin-base-ade-semantic")
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            "facebook/mask2former-swin-base-ade-semantic").to(device).eval()
        EXCLUDE = {27, 8, 147}  # mirror, windowpane, glass
        frames = sorted([f for f in os.listdir(FRAMES_DIR) if f.endswith(".jpg")])
        with torch.no_grad():
            for i, f in enumerate(frames):
                img = Image.open(str(FRAMES_DIR / f)).convert("RGB")
                inputs = proc(images=img, return_tensors="pt").to(device)
                out = model(**inputs)
                seg = proc.post_process_semantic_segmentation(
                    out, target_sizes=[img.size[::-1]])[0].cpu().numpy()
                mask = np.ones_like(seg, dtype=np.uint8) * 255
                for c in EXCLUDE:
                    mask[seg == c] = 0
                Image.fromarray(mask).save(str(MASKS_DIR / f.replace(".jpg",".png")))
                if (i+1) % 25 == 0:
                    log(f"  Masks {i+1}/{len(frames)}")
        log(f"Máscaras: {len(frames)}")
    except Exception as e:
        log(f"WARN Mask2Former falló: {e}", "WARN")

# ══════════════════════════════════════════════════════════════
# ETAPA 5: COLMAP
# ══════════════════════════════════════════════════════════════

def run_colmap():
    log("━━━ ETAPA 5: COLMAP ━━━")
    sparse = COLMAP_DIR / "sparse"
    db = COLMAP_DIR / "database.db"
    sparse.mkdir(exist_ok=True)

    # FIX v3.8: COLMAP con SIFT en GPU inicializa OpenGL/Qt y necesita un display.
    # Lo corremos bajo xvfb (pantalla virtual) para el feature_extractor.
    # FIX v3.10: el exhaustive_matcher con GPU bajo xvfb SE CUELGA en silencio
    # (no crashea, no avanza → Timeout 900s). Por eso el matcher va SIEMPRE en CPU,
    # que con ~50-80 fotos es rápido (~1-2 min) y nunca se cuelga.
    # El feature_extractor con GPU sí funciona bien (~20s), así que ese queda en GPU.
    xvfb = _has_xvfb()
    log(f"xvfb disponible: {xvfb}")
    sift_gpu = "1" if xvfb else "0"  # sin xvfb, GPU SIFT crashea → usar CPU directo

    try:
        run(["colmap","feature_extractor",
             "--database_path", str(db), "--image_path", str(FRAMES_DIR),
             "--ImageReader.single_camera","1",
             "--ImageReader.camera_model","OPENCV",
             "--SiftExtraction.use_gpu", sift_gpu],
            TIMEOUTS["colmap_feature"], "features", use_xvfb=xvfb)
        # Matcher SIEMPRE en CPU (use_gpu=0): evita el cuelgue de GPU+xvfb.
        run(["colmap","exhaustive_matcher",
             "--database_path", str(db),
             "--SiftMatching.use_gpu","0"],
            TIMEOUTS["colmap_match"], "matching-cpu")
    except (RuntimeError, Timeout) as e:
        # Fallback: reintentar TODO con SIFT en CPU (sin GPU, sin Qt).
        # Ahora atrapa también Timeout, no solo RuntimeError.
        log(f"COLMAP falló ({e}); reintentando TODO con SIFT en CPU...", "WARN")
        if db.exists(): db.unlink()
        run(["colmap","feature_extractor",
             "--database_path", str(db), "--image_path", str(FRAMES_DIR),
             "--ImageReader.single_camera","1",
             "--ImageReader.camera_model","OPENCV",
             "--SiftExtraction.use_gpu","0"],
            TIMEOUTS["colmap_feature"], "features-cpu")
        run(["colmap","exhaustive_matcher",
             "--database_path", str(db),
             "--SiftMatching.use_gpu","0"],
            TIMEOUTS["colmap_match"], "matching-cpu2")

    run(["colmap","mapper",
         "--database_path", str(db),
         "--image_path", str(FRAMES_DIR),
         "--output_path", str(sparse)],
        TIMEOUTS["colmap_mapper"], "mapper")
    s0 = sparse / "0"
    if not s0.exists():
        raise RuntimeError("COLMAP no reconstruyó. Fotos con poca textura o overlap.")
    run(["colmap","image_undistorter",
         "--image_path", str(FRAMES_DIR),
         "--input_path", str(s0),
         "--output_path", str(COLMAP_DIR),
         "--output_type","COLMAP"],
        TIMEOUTS["colmap_undistort"], "undistort")
    log("COLMAP OK")

# ══════════════════════════════════════════════════════════════
# ETAPA 6: GSPLAT TRAINING
# ══════════════════════════════════════════════════════════════

def run_gsplat(iters):
    log(f"━━━ ETAPA 6: gsplat ({iters} iter) ━━━")
    trainer = Path("/opt/gsplat-repo/examples/simple_trainer.py")
    if not trainer.exists():
        log("Clonando gsplat repo tag v1.4.0...")
        run(["git","clone","--branch","v1.4.0","--depth","1",
             "https://github.com/nerfstudio-project/gsplat.git","/opt/gsplat-repo"],
            300, "git_clone")
        reqs = "/opt/gsplat-repo/examples/requirements.txt"
        if os.path.exists(reqs):
            run(["pip","install","-r",reqs], 300, "trainer_deps")
        trainer = Path("/opt/gsplat-repo/examples/simple_trainer.py")
    run(["python", str(trainer), "default",
         "--data_dir", str(COLMAP_DIR), "--data_factor","1",
         "--result_dir", str(RESULT_DIR),
         "--max_steps", str(iters),
         "--save_steps", str(iters),
         "--eval_steps", str(iters + 1),
         "--disable_viewer"],
        TIMEOUTS["gsplat"], "gsplat")
    log("gsplat OK")

def find_ply():
    """gsplat 1.4.0 NO genera .ply: guarda un checkpoint .pt en result/ckpts/.
    Esta función localiza ese checkpoint y lo CONVIERTE a .ply estándar de
    Gaussian Splatting (el formato que abre SuperSplat, Polycam, etc.).
    """
    # 1) Si por lo que sea ya hay un .ply, úsalo.
    for root, _, files in os.walk(str(RESULT_DIR)):
        for f in files:
            if f.endswith(".ply"):
                log(f"PLY encontrado directamente: {f}")
                return os.path.join(root, f)

    # 2) Buscar el checkpoint .pt más avanzado (mayor número de step).
    ckpts = []
    for root, _, files in os.walk(str(RESULT_DIR)):
        for f in files:
            if f.endswith(".pt"):
                ckpts.append(os.path.join(root, f))
    if not ckpts:
        raise RuntimeError(
            f"gsplat no dejó ni .ply ni checkpoint .pt en {RESULT_DIR}. "
            f"El entrenamiento no guardó nada.")

    def _step_of(path):
        # nombres tipo ckpt_6999_rank0.pt → 6999
        import re
        m = re.search(r"ckpt_(\d+)", os.path.basename(path))
        return int(m.group(1)) if m else 0
    ckpt_path = sorted(ckpts, key=_step_of)[-1]
    log(f"Convirtiendo checkpoint a PLY: {os.path.basename(ckpt_path)}")

    ply_path = str(RESULT_DIR / "scene.ply")
    _convert_ckpt_to_ply(ckpt_path, ply_path)
    return ply_path


def _convert_ckpt_to_ply(ckpt_path, ply_path):
    """Lee un checkpoint de gsplat y escribe un .ply estándar de 3DGS."""
    import torch
    from plyfile import PlyData, PlyElement

    ckpt = torch.load(ckpt_path, map_location="cpu")
    splats = ckpt["splats"] if "splats" in ckpt else ckpt
    # splats es un state_dict con: means, scales, quats, opacities, sh0, shN
    def _get(name):
        if name not in splats:
            raise RuntimeError(f"Checkpoint sin campo '{name}'. Campos: {list(splats.keys())}")
        return splats[name].detach().cpu().numpy()

    means = _get("means").astype(np.float32)            # (N,3)
    scales = _get("scales").astype(np.float32)           # (N,3) en log-espacio
    quats = _get("quats").astype(np.float32)             # (N,4)
    opacities = _get("opacities").astype(np.float32).reshape(-1, 1)  # (N,1) logit
    sh0 = _get("sh0").astype(np.float32)                 # (N,1,3) DC
    shN = splats.get("shN", None)                        # (N,K,3) resto (opcional)

    N = means.shape[0]
    log(f"Gaussianos en el modelo: {N}")

    # sh0 viene como (N,1,3) → aplanar a (N,3) para f_dc_0..2
    f_dc = sh0.reshape(N, -1)                            # (N,3)
    # shN viene como (N,K,3) → aplanar a (N, K*3) en orden [coef, canal]
    if shN is not None:
        shN = shN.detach().cpu().numpy().astype(np.float32)
        f_rest = shN.reshape(N, -1)                      # (N, K*3)
    else:
        f_rest = np.zeros((N, 0), dtype=np.float32)

    # Nombres de columnas estándar de 3DGS (los que esperan los visores)
    cols = ["x", "y", "z", "nx", "ny", "nz"]
    f_dc_names = [f"f_dc_{i}" for i in range(f_dc.shape[1])]
    f_rest_names = [f"f_rest_{i}" for i in range(f_rest.shape[1])]
    cols += f_dc_names + f_rest_names
    cols += ["opacity"]
    cols += [f"scale_{i}" for i in range(scales.shape[1])]
    cols += [f"rot_{i}" for i in range(quats.shape[1])]

    normals = np.zeros((N, 3), dtype=np.float32)
    data = np.concatenate(
        [means, normals, f_dc, f_rest, opacities, scales, quats], axis=1
    ).astype(np.float32)

    dtype = [(c, "f4") for c in cols]
    verts = np.empty(N, dtype=dtype)
    for i, c in enumerate(cols):
        verts[c] = data[:, i]

    el = PlyElement.describe(verts, "vertex")
    PlyData([el], byte_order="<").write(ply_path)
    size_mb = os.path.getsize(ply_path) / 1024**2
    log(f"PLY escrito: {ply_path} ({size_mb:.1f} MB, {N} gaussianos)")

# ══════════════════════════════════════════════════════════════
# ETAPA 7: CLEANUP (gsplat ya prunea internamente)
# ══════════════════════════════════════════════════════════════

def cleanup_ply(ply_path):
    log("━━━ ETAPA 7: Cleanup ━━━")
    log("gsplat ya pruneó outliers durante training. PLY conservado.")
    return ply_path

# ══════════════════════════════════════════════════════════════
# ETAPA 8: COLLISION MESH (.glb)
# ══════════════════════════════════════════════════════════════

def gen_collision(ply_path, glb_path):
    log("━━━ ETAPA 8: Collision mesh ━━━")
    try:
        run(["splat-transform", str(ply_path), str(glb_path), "-K"],
            TIMEOUTS["collision"], "collision")
        if os.path.exists(glb_path):
            log(f"Collision mesh: {os.path.getsize(glb_path)//1024} KB")
            return True
    except Exception as e:
        log(f"WARN collision falló: {e}", "WARN")
    return False

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    global _keep_heartbeat
    log(f"=== JOB {TOUR_ID} (calidad={QUALITY}) ===")

    # Primer heartbeat: confirmar que el backend nos oye antes de gastar GPU
    # FIX BUG 2: hasta 15 intentos × 5s = 75s (suficiente para que el backend
    # commitee el job en DB después de POST /api/jobs)
    log("Primer heartbeat... (puede tardar hasta 75s si el backend está despertando)")
    backend_ok = False
    for attempt in range(15):
        if callback({"type":"progress","progress":0.0,
                     "message":"Pod arrancó, preparando..."}):
            log(f"Backend OK al intento {attempt+1}")
            backend_ok = True
            break
        log(f"Heartbeat falló ({attempt+1}/15), reintento en 5s...", "WARN")
        time.sleep(5)
    if not backend_ok:
        log("CRITICAL: backend no responde tras 75s. Abortando.", "ERROR")
        sys.exit(1)

    hb = threading.Thread(target=heartbeat_loop, daemon=True)
    hb.start()

    try:
        if QUALITY not in ITERS:
            raise RuntimeError(f"quality inválido: {QUALITY}")
        iters = ITERS[QUALITY]

        # Descargar ZIP
        download(INPUT_URL, INPUT_ZIP)

        # Extraer
        report(0.10, "Extrayendo ZIP...")
        with zipfile.ZipFile(INPUT_ZIP, "r") as z:
            z.extractall(RAW_DIR)
        INPUT_ZIP.unlink(missing_ok=True)

        # Aplanar subcarpetas si las hay
        for _ in range(3):
            items = os.listdir(RAW_DIR)
            if len(items) == 1 and os.path.isdir(RAW_DIR / items[0]):
                inner = RAW_DIR / items[0]
                for f in os.listdir(inner):
                    shutil.move(str(inner / f), str(RAW_DIR))
                inner.rmdir()
            else:
                break
        if (RAW_DIR / "images").is_dir():
            for f in os.listdir(RAW_DIR / "images"):
                shutil.move(str(RAW_DIR / "images" / f), str(RAW_DIR))
            (RAW_DIR / "images").rmdir()

        # ETAPA 1
        report(0.15, "Extracción de frames")
        count = extract_frames()
        if count < MIN_IMGS:
            raise RuntimeError(f"Solo {count} frames, mínimo {MIN_IMGS}")

        # ETAPA 2
        report(0.20, "Filtro de blur")
        kept = filter_blur()
        if kept < MIN_IMGS:
            raise RuntimeError(f"Solo {kept} frames nítidos, mínimo {MIN_IMGS}")

        # Limitar a MAX_IMGS
        all_frames = sorted([f for f in os.listdir(FRAMES_DIR) if f.endswith(".jpg")])
        if len(all_frames) > MAX_IMGS:
            step = len(all_frames) / MAX_IMGS
            keep = {int(i * step) for i in range(MAX_IMGS)}
            for i, f in enumerate(all_frames):
                if i not in keep:
                    os.remove(str(FRAMES_DIR / f))
        final_count = len([f for f in os.listdir(FRAMES_DIR) if f.endswith(".jpg")])

        # ETAPA 3
        report(0.30, "Depth Anything V2")
        gen_depth()

        # ETAPA 4
        report(0.40, "Mask2Former")
        gen_masks()

        # ETAPA 5
        report(0.50, "COLMAP")
        run_colmap()

        # ETAPA 6
        report(0.65, f"gsplat training ({iters} iter)")
        run_gsplat(iters)

        ply_path = find_ply()
        ply_mb = os.path.getsize(ply_path) / (1024 * 1024)
        log(f"PLY: {ply_mb:.1f} MB")

        # ETAPA 7
        report(0.93, "Cleanup")
        ply_path = cleanup_ply(ply_path)

        # ETAPA 8
        report(0.95, "Collision mesh")
        glb_path = str(RESULT_DIR / "collision.glb")
        has_glb = gen_collision(ply_path, glb_path)

        # Subir
        report(0.97, "Subiendo .ply a R2...")
        upload(ply_path, UPLOAD_URL_PLY)
        if has_glb and UPLOAD_URL_GLB:
            try:
                report(0.99, "Subiendo .glb a R2...")
                upload(glb_path, UPLOAD_URL_GLB)
            except Exception as e:
                log(f"WARN upload .glb falló: {e}", "WARN")

        _keep_heartbeat = False
        elapsed = round(time.time() - _t0, 1)

        # Callback de éxito (reintentos)
        success = {
            "type": "completed",
            "frames_total": count,
            "frames_used": final_count,
            "ply_mb": round(ply_mb, 2),
            "has_collision": has_glb,
            "seconds": elapsed,
            "quality": QUALITY,
            "iterations": iters,
        }
        for _ in range(5):
            if callback(success):
                log("Backend notificado del éxito")
                break
            time.sleep(5)
        log(f"=== SUCCESS {elapsed}s ===")

    except Exception as e:
        _keep_heartbeat = False
        traceback.print_exc()
        err = {
            "type": "error",
            "error_code": e.__class__.__name__,
            "error_message": str(e)[:500],
            "log": full_log() + "\n\nTRACEBACK:\n" + traceback.format_exc(),
        }
        for _ in range(3):
            if callback(err):
                log("Backend notificado del error")
                break
            time.sleep(5)
        sys.exit(1)

if __name__ == "__main__":
    main()
