#!/usr/bin/env python3
"""
Gaussian Worker v4 — MALLA (OpenMVS) Edition
═══════════════════════════════════════════════════════════════════
Genera una MALLA TEXTURIZADA (tipo Polycam) en vez de un splat.
Corre dentro de un POD RunPod/Vast usando la imagen propia
'felipegil0106/gaussian-mesh:v1' que trae COLMAP + OpenMVS precompilados.

PIPELINE (malla):
  1. Extracción de frames
  2. Filtro de blur adaptativo
  (3 y 4 Depth/Masks: OMITIDAS — OpenMVS hace su propia densa)
  5. COLMAP (posiciones de cámara + undistort)
  6. OpenMVS: InterfaceCOLMAP → DensifyPointCloud → ReconstructMesh
     → RefineMesh (opcional) → TextureMesh → .obj texturizado
  7. Convertir .obj → .glb (formato para móvil/visores) y subir

CARACTERÍSTICAS:
  - Lee parámetros desde env vars (no de un dict 'job')
  - Manda callbacks HMAC al backend (progress, completed, error)
  - Heartbeat cada 30s
  - En error, manda el LOG COMPLETO al backend (descargable después)
  - El código de gsplat/depth/masks sigue presente pero NO se usa
    (por si se quiere reactivar el modo splat en el futuro)
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
    # OpenMVS (malla): DensifyPointCloud es el paso pesado
    "mvs_interface":300, "mvs_densify":2400, "mvs_mesh":1200,
    "mvs_refine":1800, "mvs_texture":1800,
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

# LOG COMPLETO: antes limitábamos a las últimas 500 líneas (deque maxlen=500),
# lo que ocultaba el inicio del proceso. Para DIAGNOSTICAR de verdad, ahora
# guardamos TODAS las líneas (lista normal, sin límite). Así se ve absolutamente
# todo lo que el worker hace, de principio a fin, y todo se guarda y se descarga.
_LOG = []
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

def mem_info(etiqueta=""):
    """Mide y registra la memoria RAM del sistema (para DIAGNÓSTICO).
    Lee /proc/meminfo (siempre disponible en Linux, sin librerías extra).
    Así sabemos EXACTAMENTE cuánta memoria hay libre en cada punto crítico,
    para confirmar o descartar si el problema es falta de memoria (OOM)."""
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for ln in f:
                parts = ln.split(":")
                if len(parts) == 2:
                    k = parts[0].strip()
                    v = parts[1].strip().split()[0]  # en kB
                    info[k] = int(v)
        total_gb = info.get("MemTotal", 0) / 1024 / 1024
        avail_gb = info.get("MemAvailable", 0) / 1024 / 1024
        usada_gb = total_gb - avail_gb
        # memoria del proceso actual (VmRSS)
        proc_gb = 0.0
        try:
            with open("/proc/self/status") as f:
                for ln in f:
                    if ln.startswith("VmRSS:"):
                        proc_gb = int(ln.split()[1]) / 1024 / 1024
                        break
        except Exception:
            pass
        log(f"[MEMORIA {etiqueta}] total={total_gb:.1f}GB usada={usada_gb:.1f}GB "
            f"libre={avail_gb:.1f}GB proceso={proc_gb:.2f}GB")
    except Exception as e:
        log(f"[MEMORIA {etiqueta}] no se pudo medir: {e}", "WARN")

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
    # El heartbeat ahora MANDA EL LOG acumulado en cada latido. Así, si el pod
    # muere de golpe (p.ej. Poisson se queda sin memoria), el backend ya tiene
    # el log hasta donde llegó, y se puede descargar completo. Antes el log solo
    # se mandaba al final → si el pod moría, el log se perdía ("Sin log").
    while _keep_heartbeat:
        try:
            callback({"type":"progress", "progress":_current_progress,
                      "message":_current_message, "log": full_log()})
        except: pass
        time.sleep(30)

# ══════════════════════════════════════════════════════════════
# UTILIDADES
# ══════════════════════════════════════════════════════════════

class Timeout(Exception): pass

def _has_xvfb():
    """Detecta si xvfb-run está disponible (pantalla virtual para COLMAP)."""
    return shutil.which("xvfb-run") is not None

def run(cmd, timeout, name="", use_xvfb=False, cwd=None):
    # FIX v3.8: COLMAP (Qt) necesita un display aunque corra headless.
    # Sin pantalla, aborta con 'QGuiApplicationPrivate::createPlatformIntegration' rc=-6.
    # xvfb-run crea una pantalla virtual en RAM y resuelve el crash.
    if use_xvfb and _has_xvfb():
        cmd = ["xvfb-run", "-a", "-s", "-screen 0 1280x1024x24"] + list(cmd)
    log(f"[{name}] " + " ".join(str(c) for c in cmd[:8]))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False, cwd=cwd)
    except subprocess.TimeoutExpired:
        raise Timeout(f"Timeout {name} ({timeout}s)")
    # LOG COMPLETO: registramos TODA la salida del comando (stdout + stderr),
    # no solo las últimas líneas. Así, ante cualquier problema, vemos el detalle
    # completo de lo que el programa dijo, no un recorte que esconde la causa.
    if r.stdout and r.stdout.strip():
        log(f"[{name}] salida:\n{r.stdout.strip()}")
    if r.stderr and r.stderr.strip():
        log(f"[{name}] mensajes:\n{r.stderr.strip()}")
    if r.returncode != 0:
        err = r.stderr or r.stdout or "(sin salida)"
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

    # ── ESTRATEGIA ANTI-OOM (rc=-9) ──
    # PROBLEMA detectado: el intento con GPU+xvfb abortaba (rc=134) y dejaba
    # memoria/procesos a medio morir; el reintento en CPU arrancaba con la RAM
    # ya comprometida y MORÍA por falta de memoria (rc=-9). Con la imagen v2
    # (más pesada) esto pasaba SIEMPRE, aunque antes funcionaba.
    # SOLUCIÓN: ir DIRECTO a CPU una sola vez, con memoria limpia y parámetros
    # livianos (resolución 1600 + 4096 features). COLMAP solo ubica las cámaras,
    # así que bajar resolución NO afecta la calidad final (la textura la pone
    # OpenMVS con las fotos completas). Esto usa MUCHA menos RAM y no se cae.
    log("COLMAP: SIFT en CPU directo (modo liviano anti-memoria)")
    run(["colmap","feature_extractor",
         "--database_path", str(db), "--image_path", str(FRAMES_DIR),
         "--ImageReader.single_camera","1",
         "--ImageReader.camera_model","OPENCV",
         "--SiftExtraction.max_num_features","4096",
         # Resolución contenida para no agotar RAM en pods con poca memoria.
         "--SiftExtraction.max_image_size","1600",
         "--SiftExtraction.use_gpu","0"],
        TIMEOUTS["colmap_feature"], "features-cpu")
    run(["colmap","exhaustive_matcher",
         "--database_path", str(db),
         "--SiftMatching.use_gpu","0"],
        TIMEOUTS["colmap_match"], "matching-cpu")

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
# ETAPA 6 (MALLA): OpenMVS  →  reemplaza a gsplat
# Cadena: InterfaceCOLMAP → DensifyPointCloud → ReconstructMesh
#         → RefineMesh (opcional) → TextureMesh → .obj texturizado
# ══════════════════════════════════════════════════════════════

def _find_first(folder, patterns):
    """Busca el primer archivo que exista entre varios nombres posibles."""
    for p in patterns:
        cand = Path(folder) / p
        if cand.exists() and cand.stat().st_size > 0:
            return cand
    return None

def run_openmvs():
    """Genera una malla texturizada con OpenMVS a partir de la salida de COLMAP.
    Los binarios viven en /usr/local/bin/OpenMVS (ya en PATH dentro de la imagen)."""
    log("━━━ ETAPA 6: OpenMVS (malla texturizada) ━━━")

    mvs_dir = RESULT_DIR / "mvs"
    mvs_dir.mkdir(parents=True, exist_ok=True)

    # COLMAP dejó images/ y sparse/ dentro de COLMAP_DIR (output_type=COLMAP).
    # InterfaceCOLMAP lee esa estructura y crea el proyecto .mvs.
    # Importante: las imágenes están en COLMAP_DIR/images.

    # ── Paso 6.1: InterfaceCOLMAP → scene.mvs ──
    log("OpenMVS 1/5: InterfaceCOLMAP (importando salida de COLMAP)...")
    run(["InterfaceCOLMAP",
         "-i", str(COLMAP_DIR),
         "-o", str(mvs_dir / "scene.mvs"),
         "--image-folder", str(COLMAP_DIR / "images")],
        TIMEOUTS["mvs_interface"], "mvs_interface")
    if not (mvs_dir / "scene.mvs").exists():
        raise RuntimeError("InterfaceCOLMAP no generó scene.mvs")

    # ── Paso 6.2: DensifyPointCloud → nube densa (USA CUDA, paso pesado) ──
    # LECCIÓN APRENDIDA: densificar al máximo (level 0) generó DEMASIADA
    # geometría en zonas sin buena foto → más agujeros negros. Volvemos a un
    # nivel intermedio que da buena densidad SIN inflar zonas mal vistas.
    #   --resolution-level 1  → buena densidad, equilibrada
    #   --number-views 4      → punto confirmado por 4 fotos
    #   --filter-point-cloud 1 → filtra puntos sueltos/ruido (clave para limpiar)
    log("OpenMVS 2/5: DensifyPointCloud (nube densa, usa GPU)...")
    run(["DensifyPointCloud",
         "scene.mvs",
         "--resolution-level", "1",
         "--min-resolution", "640",
         "--number-views", "4",
         "--filter-point-cloud", "1",
         "--fusion-mode", "0"],
        TIMEOUTS["mvs_densify"], "mvs_densify", cwd=str(mvs_dir))
    dense = _find_first(mvs_dir, ["scene_dense.mvs", "scene.mvs"])
    if dense is None:
        raise RuntimeError("DensifyPointCloud no generó nube densa")
    log(f"   nube densa: {dense.name}")

    # ── Paso 6.3: ReconstructMesh → malla de triángulos ──
    # CONTRA LAS FACETAS/HEXÁGONOS (el efecto cristalizado):
    # Las facetas son caras planas grandes en zonas mal vistas. Para reducirlas:
    #   --close-holes 100  → rellena MUCHO más con superficie continua (no deja
    #                        huecos que se vuelvan facetas planas).
    #   --smooth 5         → suaviza FUERTE (funde las facetas en superficie
    #                        continua, como el aspecto continuo de Polycam).
    #   --remove-spurious 30 → limpieza suave (no abrir huecos).
    #   SIN --decimate: simplificar creaba caras grandes planas = MÁS facetas
    #                   visibles. Dejamos la malla densa para superficie suave.
    log("OpenMVS 3/5: ReconstructMesh (malla suave, anti-facetas)...")
    # CONTRA EL MOSAICO/HEXÁGONOS (descubrimiento clave):
    # La malla salía con ~2.5 MILLONES de vértices (Polycam usa ~510K). Tantos
    # vértices = la textura se parte en miles de parches DIMINUTOS que se ven
    # como mosaico/hexágonos/triángulos. La solución es SIMPLIFICAR la malla:
    #   --decimate 0.2 → conserva ~20% de las caras → baja de 2.5M a ~500K
    #                    vértices (como Polycam) → parches GRANDES → textura
    #                    continua y nítida (no mosaico).
    #   --close-holes 100 / --remove-spurious 30 / --smooth 5 → como antes.
    # CONTRA EL MOSAICO/HEXÁGONOS pero SIN lavar el detalle:
    # decimate 0.2 (conservar 20%) quitaba el mosaico pero dejaba la textura
    # "deslavada"/borrosa (quitaba demasiada geometría). decimate 0.5
    # (conservar 50%) es el PUNTO MEDIO: reduce el mosaico y conserva detalle.
    #   --decimate 0.5 → de ~2.5M a ~1.2M vértices (entre el exceso y el lavado)
    run(["ReconstructMesh", dense.name,
         "--close-holes", "100",
         "--remove-spurious", "30",
         "--decimate", "0.5",
         "--smooth", "3"],
        TIMEOUTS["mvs_mesh"], "mvs_mesh", cwd=str(mvs_dir))
    mesh = _find_first(mvs_dir, [
        "scene_dense_mesh.ply", "scene_mesh.ply", "scene_dense.ply"])
    if mesh is None:
        raise RuntimeError("ReconstructMesh no generó la malla")
    log(f"   malla cruda: {mesh.name}")
    try:
        import trimesh as _tm
        _m = _tm.load(str(mesh), process=False)
        nv = len(_m.vertices) if hasattr(_m, "vertices") else 0
        log(f"   malla: ~{nv:,} vértices")
    except Exception:
        pass

    # ── Paso 6.3b: SUPERFICIE POISSON (open3d) — anti-triángulos de RAÍZ ──
    # CAMBIO DE FONDO: la malla de OpenMVS, por su naturaleza, queda irregular
    # y "facetada" (se ven triángulos/hexágonos al acercar). Poisson toma los
    # puntos de esa malla y reconstruye una SUPERFICIE CERRADA Y CONTINUA:
    # rellena huecos (paredes que faltaban), redondea formas (una almohada se
    # ve redonda), y elimina las facetas sueltas. Es gratis y libre comercial.
    # Poisson DEBE funcionar (es la mejora anti-triángulos). NO lo abandonamos:
    # lo hicimos rápido (reutilizar normales de OpenMVS + 80k puntos + depth 8),
    # así corre en pocos minutos. El heartbeat de fondo (cada 30s) mantiene vivo
    # al pod mientras Poisson trabaja, así que el backend no lo mata por timeout.
    # ✅ VOLVIMOS A POISSON (USAR_POISSON = True), como en el render que SÍ
    # gustó. La Opción 3 (apagar Poisson) trajo de vuelta triángulos, ruido de
    # colores y deformación, porque Poisson era justo el paso que suaviza la
    # superficie, la cierra/limpia y quita los puntos sueltos. Confirmado:
    # Poisson es el que da la suavidad. El color lo atacamos por otro lado.
    USAR_POISSON = True
    if USAR_POISSON:
        try:
            log("OpenMVS 3b/5: Reconstrucción de superficie Poisson (anti-facetas)...")
            # open3d ya viene en la imagen v2. Si no estuviera, intento instalarlo.
            try:
                import open3d as o3d
            except Exception:
                log("   instalando open3d (necesario para Poisson)...")
                _pip_install(["open3d"], intentos=3, timeout=900, obligatorio=False)
                import open3d as o3d
            import numpy as _npp

            # Heartbeat explícito ANTES de empezar.
            callback({"type":"progress", "progress":_current_progress,
                      "message":"Poisson: preparando nube...", "log": full_log()})
            mem_info("inicio-Poisson")
            log("   [Poisson] leyendo malla de OpenMVS...")
            _t_p = time.time()
            _src = o3d.io.read_triangle_mesh(str(mesh))
            _src.compute_vertex_normals()
            log(f"   [Poisson] malla leída en {time.time()-_t_p:.1f}s")
            _pcd = o3d.geometry.PointCloud()
            _pcd.points = _src.vertices
            # CLAVE: la malla de OpenMVS YA tiene normales por vértice. Las
            # reutilizamos (rápido) en vez de recalcularlas con estimate_normals,
            # que con 150k puntos en el pod tardaba >25 min y se colgaba.
            if _src.has_vertex_normals():
                _pcd.normals = _src.vertex_normals
                log("   [Poisson] reutilizando normales de OpenMVS (rápido)")
            else:
                log("   [Poisson] la malla NO trae normales; habrá que estimarlas")
            _n_pts = len(_pcd.points)
            log(f"   nube de entrada: {_n_pts:,} puntos")
            mem_info("nube-cargada")
            # ── VELOCIDAD Y MEMORIA ──
            # El pod es MÁS LENTO que mi entorno. Con 150k se colgaba. Bajamos a
            # 80k: superficie suave de buena calidad, rápido y sin saturar
            # memoria. Selección aleatoria precisa para respetar el objetivo.
            _LIMITE_SEGURO = 80000
            if _n_pts > _LIMITE_SEGURO:
                _idx = _npp.random.choice(_n_pts, _LIMITE_SEGURO, replace=False)
                _pcd = _pcd.select_by_index(list(_idx))
                log(f"   nube ajustada a {len(_pcd.points):,} puntos (rápido y seguro)")
            mem_info("nube-ajustada")
            # ── ARREGLO DE NORMALES (la causa real del cuelgue) ──
            # El diagnóstico confirmó: NO era memoria (921 GB libres). Poisson
            # se colgaba con la nube real. Causa probable: las normales que
            # reutilizamos de OpenMVS estaban en mal estado (sin orientación
            # coherente, o con valores raros) → Poisson entra en bucle infinito.
            # SOLUCIÓN: siempre re-estimar Y orientar las normales de forma
            # coherente antes de Poisson. Con 80k puntos esto es rápido.
            callback({"type":"progress", "progress":_current_progress,
                      "message":"Poisson: preparando normales...", "log": full_log()})
            log("   [Poisson] re-estimando normales limpias (knn=18)...")
            _t_n = time.time()
            _pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamKNN(knn=18))
            # Orientar las normales de forma coherente (todas hacia afuera).
            # ESTE paso es el que faltaba: sin él, Poisson no sabe qué lado es
            # "dentro" y "fuera" y se atasca. Usamos el método liviano (hacia
            # un punto), NO el pesado (tangent_plane) que reventaba antes.
            log("   [Poisson] orientando normales (clave para que NO se cuelgue)...")
            try:
                _pcd.orient_normals_towards_camera_location(_pcd.get_center())
            except Exception as _e:
                log(f"   [Poisson] aviso al orientar: {_e}", "WARN")
            _pcd.normalize_normals()
            # Verificar que NO haya normales inválidas (NaN), que cuelgan Poisson.
            _nn = _npp.asarray(_pcd.normals)
            _malas = _npp.isnan(_nn).any(axis=1)
            if _malas.any():
                log(f"   [Poisson] {int(_malas.sum())} normales inválidas (NaN) → limpiando")
                _buenas = ~_malas
                _pcd = _pcd.select_by_index(list(_npp.where(_buenas)[0]))
            log(f"   [Poisson] normales listas en {time.time()-_t_n:.1f}s "
                f"({len(_pcd.points):,} puntos válidos)")
            mem_info("normales-listas")
            callback({"type":"progress", "progress":_current_progress,
                      "message":"Poisson: reconstruyendo superficie...", "log": full_log()})
            # depth 8: más rápido y liviano que 9, superficie igual de suave
            # para una escena de cuarto.
            log("   [Poisson] INICIANDO create_from_point_cloud_poisson (depth=8)...")
            log("   [Poisson] (si se traba, es AQUÍ; mira la memoria de abajo)")
            mem_info("antes-de-poisson")
            _t_pois = time.time()
            # ── RED DE SEGURIDAD (timeout real) ──
            # Si las normales arregladas resuelven el cuelgue → Poisson termina
            # en segundos. PERO por si acaso, lo corremos con un límite de 20 min.
            # Si se pasa, ABANDONAMOS Poisson y seguimos con la malla de OpenMVS,
            # para que el render TERMINE pase lo que pase. IMPORTANTE: esto NO
            # corta el pod — solo deja Poisson de lado y continúa el render
            # normal con la malla de OpenMVS. El pod sigue vivo y entrega .glb.
            _pois_out = {}
            def _correr_poisson():
                _m, _d = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    _pcd, depth=8, scale=1.1, linear_fit=False)
                _pois_out["mesh"] = _m
                _pois_out["dens"] = _d
            _th_pois = threading.Thread(target=_correr_poisson, daemon=True)
            _th_pois.start()
            _th_pois.join(timeout=1200)  # 20 minutos de límite de seguridad
            if _th_pois.is_alive():
                # Poisson se colgó otra vez → abandonar y usar OpenMVS.
                log("   [Poisson] NO terminó en 20 min → abandono y sigo con OpenMVS", "WARN")
                mem_info("poisson-timeout")
                raise TimeoutError("Poisson excedió 20 min")
            _pm = _pois_out["mesh"]
            _dens = _pois_out["dens"]
            log(f"   [Poisson] create_from_point_cloud_poisson TERMINÓ en {time.time()-_t_pois:.1f}s")
            mem_info("despues-de-poisson")
            callback({"type":"progress", "progress":_current_progress,
                      "message":"Poisson: limpiando y suavizando...", "log": full_log()})
            # Recortar las zonas "infladas" de baja densidad (artefactos Poisson)
            _dens = _npp.asarray(_dens)
            _pm.remove_vertices_by_mask(_dens < _npp.quantile(_dens, 0.05))
            log(f"   [Poisson] tras recortar baja densidad: {len(_pm.vertices):,} vértices")
            # Si quedó muy densa, simplificar a ~600K caras (como Polycam).
            if len(_pm.triangles) > 600000:
                log("   [Poisson] simplificando a 600k caras...")
                _pm = _pm.simplify_quadric_decimation(600000)
            # Suavizado final ligero (superficie tersa)
            _pm = _pm.filter_smooth_simple(number_of_iterations=2)
            _pm.compute_vertex_normals()
            mem_info("fin-Poisson")
            _poisson_path = mvs_dir / "scene_poisson.ply"
            o3d.io.write_triangle_mesh(str(_poisson_path), _pm)
            if _poisson_path.exists() and len(_pm.vertices) > 1000:
                mesh = _poisson_path
                log(f"   ✓ Superficie Poisson: {len(_pm.vertices):,} vértices, "
                    f"{len(_pm.triangles):,} caras (continua, sin facetas)")
            else:
                # Poisson NO produjo malla válida → CORTAR el job (no seguir).
                log("   [Poisson] no produjo malla válida → CORTANDO el render", "ERROR")
                raise RuntimeError("Poisson no produjo malla válida")
        except Exception as e:
            # CAMBIO solicitado: si Poisson falla, CORTAMOS el job con error
            # (NO seguimos con OpenMVS). Así no se gasta GPU texturizando una
            # malla facetada que no queremos, y el log completo queda guardado
            # para diagnosticar exactamente qué pasó con Poisson.
            log(f"   [Poisson] FALLÓ: {e}", "ERROR")
            log("   [Poisson] CORTANDO el render (no se sigue con OpenMVS, por decisión)", "ERROR")
            raise RuntimeError(f"Poisson falló y el render se detuvo: {e}")

    # ── Paso 6.4: RefineMesh → suaviza/mejora ──
    # OPTIMIZACIÓN: desactivado por defecto. En pruebas tardaba ~4.5 min y NO
    # dejaba archivo útil (la malla cruda ya es buena para nuestro caso).
    # Para reactivarlo, pon USAR_REFINE_MESH = True abajo.
    USAR_REFINE_MESH = False
    refined = mesh
    if USAR_REFINE_MESH:
        try:
            log("OpenMVS 4/5: RefineMesh (mejorando malla, opcional)...")
            run(["RefineMesh", dense.name,
                 "-m", mesh.name,
                 "--resolution-level", "1"],
                TIMEOUTS["mvs_refine"], "mvs_refine", cwd=str(mvs_dir))
            r = _find_first(mvs_dir, [
                mesh.stem + "_refine.ply", "scene_dense_mesh_refine.ply"])
            if r is not None:
                refined = r
                log(f"   malla refinada: {refined.name}")
            else:
                log("   RefineMesh no dejó archivo nuevo; uso la malla cruda", "WARN")
        except (RuntimeError, Timeout) as e:
            log(f"   RefineMesh falló ({e}); sigo con la malla sin refinar", "WARN")
    else:
        log("OpenMVS 4/5: RefineMesh OMITIDO (optimización; la malla cruda basta)")

    # ── Paso 6.5: TextureMesh → pega las fotos sobre la malla → .obj final ──
    # RECONSTRUCCIÓN (la clave del COLOR, estudiando Polycam):
    #   Polycam NUNCA deja negro: si una cara no se ve bien, usa color vecino.
    #   OpenMVS por defecto deja las zonas sin foto en NEGRO (de ahí el problema).
    # SOLUCIÓN: --empty-color con un GRIS claro (no negro). El valor es un entero
    #   0xRRGGBB; usamos 0xBEBEBE (gris claro ~190) para que las zonas sin foto
    #   se vean GRISES neutras, no negras. Así el render NUNCA se ve negro.
    #   --max-texture-size 4096 → archivo liviano que carga bien.
    #
    # ⚠️ SEAM LEVELING DESACTIVADO (causa del crash std::out_of_range):
    #   El seam-leveling (igualar color entre parches) de OpenMVS CRASHEA con
    #   mallas que NO salieron de su propio ReconstructMesh. Nuestra malla viene
    #   de Poisson (open3d), que no trae los datos de visibilidad por-vértice que
    #   el seam-leveling necesita → intenta leer una llave que no existe en su
    #   mapa interno → std::out_of_range → rc=-6 → crash en el último paso.
    #   Poniendo global/local seam-leveling en 0, OpenMVS se salta ese paso y
    #   genera la textura igual (puede que con costuras de color algo visibles,
    #   pero el render TERMINA y sale el .glb). La calidad de costuras se puede
    #   recuperar después; ahora la prioridad es obtener el archivo final.
    # ⚠️ SEAM LEVELING APAGADO (CONFIRMADO 2 veces que crashea con Poisson).
    #   Probamos encenderlo con la malla limpia (Opción 2) y AÚN crashea
    #   (std::out_of_range): el seam-leveling necesita datos de visibilidad
    #   por-vértice que la malla de Poisson NO tiene, por más limpia que esté.
    #   Conclusión: seam-leveling + Poisson son incompatibles. Va en 0 (original
    #   que SÍ funcionó). El brillo lo arreglamos en post-proceso (Opción 1).
    empty_gris = str(0xBEBEBE)  # gris claro en decimal = 12500670
    log("OpenMVS 5/5: TextureMesh (color + relleno gris, NO negro)...")
    # CONTRA EL EFECTO "DESLAVADO"/BORROSO:
    # Bajamos el suavizado de textura para conservar más nitidez por zona.
    # NOTA: textura en 4096 (NO 8192). Con 8192, LaMa intentaba cargar la
    # imagen completa en la GPU y pedía ~49 GB → CUDA out of memory → crash.
    # 4096 es buena resolución y LaMa la aguanta sin quedarse sin memoria.
    run(["TextureMesh", dense.name,
         "-m", refined.name,
         "--export-type", "obj",
         "--max-texture-size", "4096",
         "--global-seam-leveling", "0",
         "--local-seam-leveling", "0",
         "--cost-smoothness-ratio", "0.1",
         "--empty-color", empty_gris,
         "-o", "scene_textured.obj"],
        TIMEOUTS["mvs_texture"], "mvs_texture", cwd=str(mvs_dir))

    # Diagnóstico: listar TODO lo que generó TextureMesh (clave para depurar)
    try:
        generados = sorted(os.listdir(str(mvs_dir)))
        log(f"   TextureMesh generó: {generados}")
    except Exception:
        pass

    textured = _find_first(mvs_dir, [
        "scene_textured.obj", refined.stem + "_texture.obj",
        "scene_dense_mesh_refine_texture.obj", "scene_dense_mesh_texture.obj"])
    if textured is None:
        raise RuntimeError("TextureMesh no generó el .obj texturizado")
    log(f"OpenMVS OK → malla texturizada: {textured.name}")
    return str(textured)

# ══════════════════════════════════════════════════════════════
# ETAPA 6.6 (MALLA): IA para relleno natural (LaMa) y nitidez (Real-ESRGAN)
# ══════════════════════════════════════════════════════════════
# Estas dos IAs imitan lo que hace Polycam:
#   - LaMa: rellena las zonas sin foto de forma NATURAL y continua (mata las
#     facetas/hexágonos), en vez del relleno borroso casero.
#   - Real-ESRGAN: sube la nitidez de la textura (súper-resolución con IA).
# COMPORTAMIENTO PEDIDO: si la IA falla (instalación o ejecución), el worker
# PARA TODO de inmediato y manda el log del error al backend (para descargarlo),
# en vez de seguir con el método casero. Así se ataca el problema apenas surge.
# Para volver al modo casero sin IA, pon USAR_IA = False.

USAR_IA = True  # True = usar IA y PARAR si falla; False = método casero sin IA

# Cache global de modelos (para no recargarlos por cada textura)
_LAMA_MODEL = None
_ESRGAN_MODEL = None


def _pip_install(args, intentos=3, timeout=600, obligatorio=True):
    """Instala con pip REINTENTANDO si falla. Los fallos de instalación de las
    IAs suelen ser TEMPORALES (corte de red o PyPI saturado), no del comando.
    Reintentar 3 veces hace el arranque resistente a esos cortes.
    Si obligatorio=False, no lanza excepción aunque falle (deps opcionales)."""
    ultimo = None
    for n in range(1, intentos + 1):
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "--quiet"]
                           + args, check=True, timeout=timeout)
            return True
        except Exception as e:
            ultimo = e
            if n < intentos:
                log(f"   IA: intento {n}/{intentos} falló; reintentando en 5s...", "WARN")
                time.sleep(5)
    if obligatorio:
        raise RuntimeError(f"pip install {' '.join(args)} falló tras {intentos} intentos: {ultimo}")
    return False


def _parche_basicsr_torchvision():
    """basicsr (que usa Real-ESRGAN) importa 'torchvision.transforms.
    functional_tensor', un módulo que las versiones nuevas de torchvision YA
    eliminaron → ModuleNotFoundError → realesrgan no se puede usar.
    Este parche crea ese módulo apuntando al nuevo 'functional', ANTES de
    importar basicsr/realesrgan. Es la solución estándar de la comunidad."""
    import sys as _sys, types as _types
    if "torchvision.transforms.functional_tensor" in _sys.modules:
        return
    try:
        import torchvision.transforms.functional as _F
        _mod = _types.ModuleType("torchvision.transforms.functional_tensor")
        # basicsr usa rgb_to_grayscale; redirigir al nuevo módulo
        _mod.rgb_to_grayscale = _F.rgb_to_grayscale
        _sys.modules["torchvision.transforms.functional_tensor"] = _mod
    except Exception:
        pass


def _instalar_dependencias_ia():
    """Instala (si faltan) las librerías de IA. Se llama una sola vez.
    Si algo NO se puede instalar, LANZA una excepción (el worker parará y
    mandará el log, según lo pedido: atacar el problema apenas surja).
    No forzamos versión de torch si la imagen ya lo trae (para no romper CUDA)."""
    global _LAMA_MODEL, _ESRGAN_MODEL

    tiene_torch = False
    try:
        import torch  # noqa
        tiene_torch = True
    except Exception:
        log("   IA: torch no está; intentando instalar (puede tardar 1-3 min)...")
        ultimo_error = None
        for args in (
            ["torch", "torchvision", "--index-url",
             "https://download.pytorch.org/whl/cu121"],
            ["torch", "torchvision"],
        ):
            if _pip_install(args, intentos=2, timeout=1200, obligatorio=False):
                tiene_torch = True
                break
            else:
                ultimo_error = "instalación falló tras reintentos"
                log(f"   IA: intento de instalar torch falló ({args[0]})", "WARN")
        if not tiene_torch:
            raise RuntimeError(f"No se pudo instalar torch para la IA: {ultimo_error}")

    # LaMa (relleno natural). Instalar SIN dependencias: simple-lama exige
    # pillow<10 y pip intentaba COMPILAR un Pillow viejo desde cero, lo que
    # FALLA en el pod (exit 1). Con --no-deps usa el torch/pillow/numpy que el
    # pod YA tiene. Con reintentos por si la descarga falla por red.
    try:
        import simple_lama_inpainting  # noqa
    except Exception:
        log("   IA: instalando simple-lama-inpainting (sin tocar dependencias)...")
        _pip_install(["--no-deps", "simple-lama-inpainting"], obligatorio=True)
        # 'fire' es la única dependencia liviana que LaMa usa y que --no-deps
        # no trae; el resto (torch, cv2, numpy, PIL) ya está en el pod.
        _pip_install(["fire"], obligatorio=False)
        try:
            import simple_lama_inpainting  # noqa
        except Exception as e:
            raise RuntimeError(f"No se pudo importar LaMa tras instalar: {e}")

    # Real-ESRGAN (nitidez). Instalar SIN tocar torch (--no-deps) y traer sus
    # dependencias reales aparte. basicsr da un import roto con torchvision
    # nuevo, así que aplicamos el parche ANTES de importarlo.
    log("   IA: preparando Real-ESRGAN (con parche de torchvision)...")
    # basicsr y realesrgan sin arrastrar torch (con reintentos por red)
    _pip_install(["--no-deps", "basicsr", "realesrgan"], obligatorio=False)
    # dependencias livianas que basicsr/realesrgan necesitan (SIN torch).
    # Las instalamos TODAS de una vez para evitar el goteo de errores
    # (scipy, tqdm, etc. van apareciendo uno por uno si faltan).
    for dep in ("addict", "future", "lmdb", "pyyaml", "yapf", "scipy",
                "tqdm", "requests", "tensorboard", "einops", "opencv-python",
                "tb-nightly", "gfpgan", "facexlib"):
        _pip_install(["--no-deps", dep], intentos=2, timeout=300, obligatorio=False)
    # Aplicar el parche y verificar que ahora sí importa
    _parche_basicsr_torchvision()
    try:
        from realesrgan import RealESRGANer  # noqa
        from basicsr.archs.rrdbnet_arch import RRDBNet  # noqa
    except Exception as e:
        raise RuntimeError(f"No se pudo importar Real-ESRGAN tras el parche: {e}")
    return True


def aplicar_lama(img_bgr, mask):
    """Rellena con LaMa (IA) las zonas marcadas en 'mask' (255 = rellenar).
    Recibe y devuelve imagen BGR (de cv2).
    Si LaMa falla, LANZA una excepción (el worker parará y mandará el log)."""
    global _LAMA_MODEL
    import numpy as np
    from PIL import Image
    if _LAMA_MODEL is None:
        from simple_lama_inpainting import SimpleLama
        _LAMA_MODEL = SimpleLama()  # descarga el modelo la 1ª vez
    rgb = Image.fromarray(img_bgr[:, :, ::-1])
    m = Image.fromarray(mask).convert("L")
    try:
        out = _LAMA_MODEL(rgb, m)  # devuelve PIL RGB rellenado
    except RuntimeError as e:
        # Si la GPU se queda sin memoria (textura grande / GPU chica),
        # reintentar en CPU: más lento pero NO tumba el render.
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            log("   LaMa: sin memoria GPU; reintentando en CPU (más lento)...", "WARN")
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                _LAMA_MODEL.model = _LAMA_MODEL.model.cpu()
                _LAMA_MODEL.device = "cpu"
            except Exception:
                pass
            out = _LAMA_MODEL(rgb, m)
        else:
            raise
    out_np = np.array(out)
    return out_np[:, :, ::-1]  # de vuelta a BGR


def aplicar_esrgan(img_bgr):
    """Sube la nitidez de la imagen con Real-ESRGAN (IA). Devuelve la imagen
    mejorada (puede venir más grande).
    Si falla, LANZA una excepción (el worker parará y mandará el log)."""
    global _ESRGAN_MODEL
    if _ESRGAN_MODEL is None:
        _parche_basicsr_torchvision()  # asegurar el parche antes de importar
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        modelo = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                         num_block=23, num_grow_ch=32, scale=4)
        _ESRGAN_MODEL = RealESRGANer(
            scale=4,
            model_path=("https://github.com/xinntao/Real-ESRGAN/releases/"
                        "download/v0.1.0/RealESRGAN_x4plus.pth"),
            model=modelo, tile=512, tile_pad=10, pre_pad=0, half=True)
    salida, _ = _ESRGAN_MODEL.enhance(img_bgr, outscale=2)
    return salida


# ══════════════════════════════════════════════════════════════
# ETAPA 7 (MALLA): convertir .obj texturizado → .glb (para móvil/visores)
# ══════════════════════════════════════════════════════════════

def limpiar_puntos_colores(img_bgr):
    """Quita los 'puntos de colores' (verde/rojo/magenta neón) que COLMAP
    genera como ruido y OpenMVS pinta en la textura. CLAVE: NO toca zonas
    grandes de color real (como una cobija azul) — solo borra:
      (a) colores NEÓN imposibles en una habitación (verde/magenta/rojo puro), y
      (b) manchas ultra-saturadas PEQUEÑAS y aisladas.
    Lo borrado se rellena con el color de alrededor (inpaint)."""
    import numpy as np
    try:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        # (a) colores neón que no existen en interiores reales (el AZUL NO se toca)
        verde   = (H >= 35) & (H <= 85) & (S > 120) & (V > 90)
        magenta = (H >= 140) & (H <= 170) & (S > 120) & (V > 90)
        rojo    = ((H <= 10) | (H >= 172)) & (S > 160) & (V > 110)
        neon = (verde | magenta | rojo).astype(np.uint8) * 255
        # (b) manchas ultra-saturadas pequeñas y aisladas (puntos sueltos)
        cand = ((S > 175) & (V > 120)).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
        chicas = np.zeros_like(cand)
        for k in range(1, n):
            if stats[k, cv2.CC_STAT_AREA] < 1500:  # chico = punto; grande = objeto real
                chicas[labels == k] = 255
        mask = cv2.bitwise_or(neon, chicas)
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
        if mask.sum() == 0:
            return img_bgr
        return cv2.inpaint(img_bgr, mask, 5, cv2.INPAINT_TELEA)
    except Exception:
        return img_bgr  # si algo falla, devolver la imagen sin tocar


def realce_suave(img_bgr):
    """OPCIÓN 1 — aclara las zonas OSCURAS (sombras) sin quemar las claras.
    Es nuestro arreglo de las manchas oscuras SIN tocar el texturizado (así no
    puede causar el crash del seam-leveling). Dos pasos:
      1) Curva 'gamma' (<1) que LEVANTA las sombras: sube mucho el brillo de los
         píxeles oscuros y casi nada el de los claros → no se quema lo brillante.
      2) Contraste local suave (CLAHE) para recuperar definición.
    Trabaja solo sobre la luz (canal L de LAB), así NO satura ni cambia colores.
    """
    try:
        import cv2
        import numpy as np
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # 1) levantar sombras (gamma 0.80: aclara oscuros, respeta claros)
        gamma = 0.80
        lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)],
                       dtype=np.uint8)
        l = cv2.LUT(l, lut)
        # 2) contraste local suave (un poco más que antes: 1.5 → 2.0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    except Exception:
        return img_bgr


def rellenar_negro_texturas(obj_dir):
    """POST-PROCESO (imita a Polycam para que NO haya negro ni facetas):
    En las zonas NEGRAS (sin foto) de cada textura:
      - Si USAR_IA: usa LaMa (relleno natural) + Real-ESRGAN (nitidez). Si la IA
        falla, NO usa respaldo: deja subir el error para PARAR el job y que
        puedas descargar el log (atacar el problema apenas surja).
      - Si USAR_IA es False: usa el método casero (inpainting + borrón suave).
    Devuelve cuántas texturas procesó.
    """
    import glob
    try:
        import cv2
        import numpy as np
    except Exception as e:
        if USAR_IA:
            raise RuntimeError(f"OpenCV no disponible para el post-proceso: {e}")
        log(f"   (post-proceso no disponible: {e}; sigo sin rellenar)", "WARN")
        return 0

    # Preparar IA una sola vez (si está activada). Si falla, LANZA excepción.
    if USAR_IA:
        _instalar_dependencias_ia()
        log("   IA lista (LaMa + Real-ESRGAN); procesando texturas...")

    texturas = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        texturas.extend(glob.glob(os.path.join(obj_dir, ext)))
    procesadas = 0
    for tex_path in texturas:
        img = cv2.imread(tex_path)
        if img is None:
            continue
        # Tamaño ORIGINAL — hay que conservarlo EXACTO (el .obj/.mtl y las UV lo
        # esperan). Si cambia o el JPG sale corrupto, el .glb pierde la textura
        # (sale blanco). Guardamos una copia de respaldo por si la IA falla.
        h0, w0 = img.shape[:2]
        img_original = img.copy()
        gris = img.sum(axis=2)
        mask = (gris < 30).astype(np.uint8) * 255
        negro_pct = (mask > 0).sum() / mask.size * 100

        resultado = img

        if USAR_IA:
            # ── PASO A0: limpiar puntos de colores (ruido de COLMAP) ──
            # Quita los puntos verde/rojo/magenta neón ANTES de las IAs, para
            # que LaMa/ESRGAN no los "amplifiquen". No toca el color real.
            img = limpiar_puntos_colores(img)
            resultado = img
            # ── PASO A: relleno natural con LaMa (si hay zonas sin foto) ──
            if negro_pct >= 0.5:
                mask_d = cv2.dilate(mask, np.ones((5, 5), np.uint8))
                out = aplicar_lama(img, mask_d)  # si falla → para todo
                resultado = out
                log(f"   LaMa: {os.path.basename(tex_path)} {negro_pct:.0f}% sin foto → relleno natural (IA)")
            # ── PASO B: nitidez con Real-ESRGAN ──
            mejor = aplicar_esrgan(resultado)  # si falla → para todo
            resultado = mejor
            log(f"   Real-ESRGAN: {os.path.basename(tex_path)} → más nítida (IA)")
            # ── PASO C: realce suave de contraste (contra el 'deslavado') ──
            resultado = realce_suave(resultado)

            # ── NORMALIZAR antes de guardar (clave para que NO salga blanco) ──
            # 1) asegurar uint8 0-255 (ESRGAN puede devolver float o >255)
            resultado = np.clip(resultado, 0, 255).astype(np.uint8)
            # 2) asegurar 3 canales BGR
            if resultado.ndim == 2:
                resultado = cv2.cvtColor(resultado, cv2.COLOR_GRAY2BGR)
            if resultado.shape[2] == 4:
                resultado = cv2.cvtColor(resultado, cv2.COLOR_BGRA2BGR)
            # 3) tamaño EXACTO al original (LaMa/ESRGAN lo cambian)
            if resultado.shape[:2] != (h0, w0):
                resultado = cv2.resize(resultado, (w0, h0), interpolation=cv2.INTER_AREA)
            # 4) guardar JPG de alta calidad con el MISMO nombre
            cv2.imwrite(tex_path, resultado, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # ── VERIFICACIÓN DE SEGURIDAD: releer la textura guardada ──
            # Si quedó corrupta o de otro tamaño, restaurar la ORIGINAL (mejor
            # tener la textura sin IA que un render blanco sin textura).
            check = cv2.imread(tex_path)
            if check is None or check.shape[:2] != (h0, w0):
                cv2.imwrite(tex_path, img_original, [cv2.IMWRITE_JPEG_QUALITY, 95])
                log(f"   ⚠ textura IA quedó mal; restaurada la original (sin IA)", "WARN")
            procesadas += 1
        else:
            # ── Método casero (solo si la IA está desactivada) ──
            if negro_pct < 0.5:
                continue
            base = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
            suave = cv2.GaussianBlur(base, (51, 51), 0)
            kernel = np.ones((25, 25), np.uint8)
            grandes = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            grandes3 = cv2.cvtColor(grandes, cv2.COLOR_GRAY2BGR) > 0
            resultado = np.where(grandes3, suave, base)
            cv2.imwrite(tex_path, resultado)
            procesadas += 1
            log(f"   inpainting casero: {os.path.basename(tex_path)} {negro_pct:.0f}% → rellenado")
    return procesadas


def convert_mesh_to_glb(obj_path, glb_path):
    """Carga el .obj texturizado de OpenMVS y lo exporta como .glb con la
    textura EMBEBIDA.

    IMPORTANTE (arreglo del render BLANCO): después del post-proceso con IA,
    trimesh a veces NO logra enlazar la textura del .mtl y el material queda
    blanco. Por eso, tras cargar, FORZAMOS la textura manualmente sobre cada
    geometría (leyendo el .mtl para emparejar material↔imagen). Así el color
    NO depende de que trimesh resuelva el .mtl por su cuenta.
    """
    import trimesh
    from PIL import Image
    obj_path = os.path.abspath(obj_path)
    obj_dir = os.path.dirname(obj_path)
    obj_name = os.path.basename(obj_path)
    glb_path = os.path.abspath(glb_path)

    log(f"Convirtiendo malla a .glb: {obj_name}")

    # Listar lo que dejó OpenMVS, para diagnóstico (texturas, mtl, etc.)
    try:
        archivos = os.listdir(obj_dir)
        texturas = [f for f in archivos if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        mtls = [f for f in archivos if f.lower().endswith(".mtl")]
        log(f"   en carpeta: {len(texturas)} textura(s) {texturas[:3]}, {len(mtls)} .mtl")
    except Exception:
        pass

    # POST-PROCESO clave: rellenar/IA sobre las texturas (anti-negro, anti-facetas)
    n = rellenar_negro_texturas(obj_dir)
    if n:
        log(f"   ✓ {n} textura(s) procesadas (relleno + nitidez)")

    # Mapa material→textura leyendo el .mtl (para incrustar manualmente)
    mat_tex = {}
    try:
        for mtl_name in [f for f in os.listdir(obj_dir) if f.lower().endswith(".mtl")]:
            cur = None
            with open(os.path.join(obj_dir, mtl_name), "r", errors="ignore") as fh:
                for line in fh:
                    line = line.strip()
                    if line.startswith("newmtl"):
                        cur = line.split(maxsplit=1)[1].strip()
                    elif line.lower().startswith("map_kd") and cur:
                        tex_file = line.split(maxsplit=1)[1].strip()
                        mat_tex[cur] = os.path.join(obj_dir, os.path.basename(tex_file))
    except Exception as e:
        log(f"   (no se pudo leer .mtl para incrustar textura: {e})", "WARN")

    # Cargar DESDE la carpeta del .obj (rutas relativas del .mtl)
    cwd_anterior = os.getcwd()
    try:
        os.chdir(obj_dir)
        loaded = trimesh.load(obj_name, process=False)
    finally:
        os.chdir(cwd_anterior)

    # Normalizar a lista de geometrías
    if isinstance(loaded, trimesh.Scene):
        if len(loaded.geometry) == 0:
            raise RuntimeError("La malla cargada no tiene geometría")
        geoms = list(loaded.geometry.values())
        export_obj = loaded
    else:
        geoms = [loaded]
        export_obj = loaded

    # ── SUAVIZADO DE NORMALES (anti-TRIÁNGULOS / anti-facetas) ──
    # CLAVE para que NO se vean los triángulos/hexágonos: una malla siempre
    # está hecha de triángulos, pero si cada cara usa su propia normal ("flat
    # shading") se ve facetada. Con "smooth shading" (normales promediadas por
    # vértice) la luz fluye continua y las superficies se ven lisas y
    # redondeadas — una almohada se ve redonda, no como triángulos.
    #
    # IMPORTANTE: NO fusionamos vértices de forma agresiva, porque eso rompe
    # las costuras de la textura (la UV) y dañaría el color. En vez de eso,
    # generamos normales SUAVES por vértice a partir de las caras vecinas y se
    # las asignamos a la malla, SIN tocar la geometría ni las UV. Esto cambia
    # solo cómo se ilumina la superficie (el "sombreado"), no su forma.
    import numpy as _nps
    for _g in geoms:
        try:
            verts = _nps.asarray(_g.vertices)
            faces = _nps.asarray(_g.faces)
            if len(verts) == 0 or len(faces) == 0:
                continue
            # Normal de cada cara (triángulo)
            tris = verts[faces]
            fn = _nps.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
            ln = _nps.linalg.norm(fn, axis=1, keepdims=True)
            ln[ln == 0] = 1.0
            fn = fn / ln
            # Acumular en cada vértice la normal de TODAS sus caras vecinas
            # (esto es lo que promedia y suaviza el sombreado)
            vn = _nps.zeros_like(verts, dtype=_nps.float64)
            for k in range(3):
                _nps.add.at(vn, faces[:, k], fn)
            lvn = _nps.linalg.norm(vn, axis=1, keepdims=True)
            lvn[lvn == 0] = 1.0
            vn = vn / lvn
            # Asignar las normales suaves SIN cambiar vértices/caras/UV
            _g.vertex_normals = vn
        except Exception as e:
            log(f"   (no se pudo suavizar una geometría: {e})", "WARN")
    log(f"   ✓ Normales suavizadas (anti-triángulos) en {len(geoms)} geometría(s)")

    # ── FORZAR la textura manualmente sobre cada geometría (anti-blanco) ──
    # Si una geometría tiene UV pero su material quedó sin imagen, le pegamos
    # la textura correspondiente (la única, o por orden si hay varias).
    texturas_disponibles = sorted(set(mat_tex.values())) if mat_tex else [
        os.path.join(obj_dir, f) for f in sorted(
            [x for x in os.listdir(obj_dir)
             if x.lower().endswith((".jpg", ".jpeg", ".png"))])]
    incrustadas = 0
    for idx, g in enumerate(geoms):
        try:
            visual = getattr(g, "visual", None)
            uv = getattr(visual, "uv", None)
            if uv is None:
                continue  # sin UV no hay forma de mapear textura
            # Elegir la imagen para esta geometría
            img_path = None
            # por nombre de material si lo tenemos
            mat_name = getattr(getattr(visual, "material", None), "name", None)
            if mat_name and mat_name in mat_tex and os.path.exists(mat_tex[mat_name]):
                img_path = mat_tex[mat_name]
            elif idx < len(texturas_disponibles):
                img_path = texturas_disponibles[idx]
            elif texturas_disponibles:
                img_path = texturas_disponibles[0]
            if not img_path or not os.path.exists(img_path):
                continue
            # FIX DEFINITIVO del error "'JpegImageFile'/'PngImageFile' object
            # has no attribute '_im'" (bug de trimesh 4.x + Pillow nuevo al
            # EXPORTAR, en la llamada interna img.save()). La solución robusta:
            # recodificar la textura a PNG EN MEMORIA y reabrirla; eso entrega
            # un objeto PIL 100% materializado y con .format poblado, que
            # trimesh exporta sin tropezar.
            import numpy as _np, io as _io
            _base = Image.open(img_path).convert("RGB")
            _buf = _io.BytesIO()
            _base.save(_buf, format="PNG")
            _buf.seek(0)
            pil = Image.open(_buf)
            pil.load()  # materializa los píxeles (clave para evitar '_im')
            if not getattr(pil, "format", None):
                pil.format = "PNG"
            # Crear un material PBR nuevo con la textura pegada
            nuevo = trimesh.visual.material.PBRMaterial(
                baseColorTexture=pil,
                metallicFactor=0.0,
                roughnessFactor=1.0)
            g.visual = trimesh.visual.TextureVisuals(uv=uv, material=nuevo)
            incrustadas += 1
        except Exception as e:
            log(f"   (no se pudo incrustar textura en geom {idx}: {e})", "WARN")
    if incrustadas:
        log(f"   ✓ Textura incrustada manualmente en {incrustadas} geometría(s)")

    # Exportar a glb (incrusta geometría + textura en un solo archivo).
    # RED DE SEGURIDAD: si trimesh choca con el bug '_im' de Pillow al
    # exportar, recodificamos TODAS las imágenes de la escena a PNG limpio
    # (en memoria) y reintentamos. Esto cubre el caso de imágenes que trimesh
    # cargó del .mtl por su cuenta (no por nuestro incrustado manual).
    def _sanear_imagenes_de_escena(obj_para_exportar):
        import io as _io2
        geos = (obj_para_exportar.geometry.values()
                if hasattr(obj_para_exportar, "geometry") else [obj_para_exportar])
        for _g in geos:
            try:
                _mat = getattr(getattr(_g, "visual", None), "material", None)
                if _mat is None:
                    continue
                _img = getattr(_mat, "baseColorTexture", None)
                if _img is None:
                    continue
                _b = _io2.BytesIO()
                _img.convert("RGB").save(_b, format="PNG")
                _b.seek(0)
                _limpia = Image.open(_b)
                _limpia.load()
                if not getattr(_limpia, "format", None):
                    _limpia.format = "PNG"
                _mat.baseColorTexture = _limpia
            except Exception:
                pass

    try:
        export_obj.export(glb_path, file_type="glb")
    except Exception as e:
        log(f"   export falló ({e}); saneando imágenes y reintentando...", "WARN")
        _sanear_imagenes_de_escena(export_obj)
        export_obj.export(glb_path, file_type="glb")
    mb = os.path.getsize(glb_path) / (1024 * 1024)
    log(f"GLB generado: {mb:.1f} MB")

    # Verificar si la textura quedó incrustada (diagnóstico en el log)
    try:
        check = trimesh.load(glb_path)
        geoms_c = check.geometry.values() if hasattr(check, "geometry") else [check]
        tiene_tex = False
        for g in geoms_c:
            mat = getattr(getattr(g, "visual", None), "material", None)
            if mat is not None:
                img = getattr(mat, "image", None) or getattr(mat, "baseColorTexture", None)
                if img is not None:
                    tiene_tex = True
        if tiene_tex:
            log("   ✓ Textura incrustada correctamente en el .glb")
        else:
            log("   ⚠ El .glb NO tiene textura incrustada (malla saldrá sin color)", "WARN")
    except Exception as e:
        log(f"   (no se pudo verificar textura: {e})", "WARN")

    return glb_path

# ══════════════════════════════════════════════════════════════
# ETAPA 6 (SPLAT — LEGADO): GSPLAT TRAINING (ya no se usa por defecto)
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

        # ETAPA 3 y 4 (Depth, Masks): NO se usan para la malla.
        # OpenMVS calcula su propia profundidad densa, así que las saltamos.
        # Esto ADEMÁS acelera el proceso (menos pasos).
        # (El código de gen_depth/gen_masks sigue en el archivo por si algún
        #  día se reactivan, pero aquí no se llaman.)
        log("Etapas Depth/Masks omitidas (no se requieren para malla)")

        # ETAPA 5: COLMAP (ubica las cámaras — base para OpenMVS)
        report(0.45, "COLMAP (posiciones de cámara)")
        run_colmap()

        # ETAPA 6: OpenMVS (genera la malla texturizada)
        report(0.60, "OpenMVS (construyendo malla)")
        obj_path = run_openmvs()

        # ETAPA 7: convertir la malla a .glb
        report(0.92, "Convirtiendo malla a .glb")
        glb_path = str(RESULT_DIR / "scene.glb")
        convert_mesh_to_glb(obj_path, glb_path)
        glb_mb = os.path.getsize(glb_path) / (1024 * 1024)
        log(f"GLB final: {glb_mb:.1f} MB")

        # Subir la malla .glb (este es ahora el entregable principal)
        report(0.97, "Subiendo malla .glb...")
        # El backend manda UPLOAD_URL_GLB para la malla. Si por compatibilidad
        # solo viene UPLOAD_URL_PLY, subimos el .glb por esa URL igual.
        upload_url = UPLOAD_URL_GLB if UPLOAD_URL_GLB else UPLOAD_URL_PLY
        upload(glb_path, upload_url)
        has_glb = True

        _keep_heartbeat = False
        elapsed = round(time.time() - _t0, 1)

        # Callback de éxito (reintentos)
        success = {
            "type": "completed",
            "frames_total": count,
            "frames_used": final_count,
            "glb_mb": round(glb_mb, 2),
            "mesh": True,
            "has_collision": has_glb,
            "seconds": elapsed,
            "quality": QUALITY,
            # Enviar el LOG completo también en éxito, para poder revisarlo
            # después aunque la GPU ya se haya apagado.
            "log": full_log(),
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
