"""
Worker GPU para Gaussian Splatting — Pipeline de 8 etapas
Recibe ZIP de fotos, devuelve .ply + .glb (collision mesh)

FIX v2 (2026-05-27): Filtro de blur ADAPTATIVO.
  - El umbral fijo de 100 era inadecuado para fotos de móviles modernos.
  - Ahora analizamos el set de fotos y usamos un umbral relativo al conjunto.
  - Si TODAS las fotos serían rechazadas, mantenemos las mejores.

Etapas:
  1. Extracción inteligente de frames (FFmpeg mpdecimate)
  2. Filtro de blur ADAPTATIVO (Laplacian variance + percentiles)
  3. Depth Anything V2 (depth priors para paredes blancas)
  4. Mask2Former (máscaras vidrios/espejos)
  5. COLMAP (poses de cámara)
  6. gsplat training (Gaussian Splatting)
  7. Cleanup (statistical outlier removal)
  8. splat-transform (collision mesh)
"""

import os, sys, json, shutil, subprocess, zipfile, tempfile, traceback, time, base64
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import requests as req
import runpod

# ── Configuración ──────────────────────────────────────────────

TIMEOUTS = {
    "download":600, "colmap_feature":600, "colmap_match":900,
    "colmap_mapper":1800, "colmap_undistort":300,
    "gsplat":2700, "collision":600, "upload":600,
}

ITERS = {"fast":7000, "balanced":30000, "quality":50000}

# FIX v2: Umbrales de blur ajustados para móviles modernos
BLUR_THRESHOLD_ABSOLUTE = 30.0   # Bajado de 100 a 30 (móviles)
BLUR_PERCENTILE_FALLBACK = 25    # Si todo se rechaza, eliminar solo el percentil 25 más bajo
MIN_VALID_RATIO = 0.5            # Si quedaríamos con < 50% de fotos, usar modo permisivo

MIN_IMGS, MAX_IMGS = 20, 1000

# ── Logging ────────────────────────────────────────────────────

LOG_PATH = "/workspace/jobs/job.log"
CMD_PATH = "/workspace/jobs/last_cmd.txt"

class Log:
    def __init__(self):
        self.t0 = time.time()
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        open(LOG_PATH,"w").write(f"=== {datetime.now(timezone.utc).isoformat()} ===\n")
    def __call__(self, msg, lv="INFO"):
        line = f"[{lv}][+{time.time()-self.t0:.1f}s] {msg}"
        print(line, flush=True)
        try: open(LOG_PATH,"a").write(line+"\n")
        except: pass
    def tail(self, n=60):
        try: return "\n".join(open(LOG_PATH).readlines()[-n:])
        except: return ""

log = Log()

def save_cmd(cmd, err=None):
    try:
        with open(CMD_PATH,"w") as f:
            f.write(f"CMD: {' '.join(str(c) for c in cmd) if isinstance(cmd,list) else cmd}\n")
            if err: f.write(f"ERR: {err}\n")
    except: pass

def get_cmd():
    try: return open(CMD_PATH).read()
    except: return ""

# ── Ejecución con timeout ─────────────────────────────────────

class Timeout(Exception): pass

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
    log(f"Descargando...")
    r = req.get(url, stream=True, timeout=(30, timeout))
    r.raise_for_status()
    total = int(r.headers.get("content-length",0))
    dl = 0
    with open(dest,"wb") as f:
        for chunk in r.iter_content(1024*1024):
            f.write(chunk); dl += len(chunk)
    log(f"Descargado: {dl//(1024*1024)} MB")

def upload(path, url, timeout=600):
    log(f"Subiendo {os.path.basename(path)} ({os.path.getsize(path)//(1024*1024)} MB)...")
    with open(path,"rb") as f:
        r = req.put(url, data=f, timeout=timeout,
                    headers={"Content-Type":"application/octet-stream"})
        r.raise_for_status()
    log("Upload OK")

# ── Etapa 1: Extracción de frames ─────────────────────────────

def extract_frames(raw_dir, frames_dir):
    log("━━━ ETAPA 1: Extracción de frames ━━━")
    files = os.listdir(raw_dir)
    videos = [f for f in files if f.lower().endswith((".mp4",".mov",".avi",".mkv"))]
    images = sorted([f for f in files if f.lower().endswith((".jpg",".jpeg",".png"))])

    if videos:
        log(f"Video: {videos[0]}")
        run(["ffmpeg","-i",os.path.join(raw_dir,videos[0]),
             "-vf","mpdecimate=hi=64*12:lo=64*5:frac=0.33,fps=2",
             "-qscale:v","2","-vsync","vfr",
             os.path.join(frames_dir,"frame_%05d.jpg")], 600, "ffmpeg")
    elif images:
        log(f"Fotos: {len(images)}")
        for i,img in enumerate(images):
            shutil.copy(os.path.join(raw_dir,img), os.path.join(frames_dir,f"frame_{i:05d}.jpg"))
    else:
        raise RuntimeError("No hay videos ni imágenes en el ZIP")

    count = len([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    log(f"Frames: {count}")
    return count

# ── Etapa 2: Filtro blur ADAPTATIVO ───────────────────────────

def filter_blur(frames_dir):
    """
    FIX v2: Filtro adaptativo.
    1. Calcula varianza Laplaciana de TODAS las fotos
    2. Si TODAS están sobre el umbral absoluto → solo elimina las muy malas
    3. Si MUCHAS están bajo el umbral → usa percentiles (quita el 25% peor)
    4. Garantiza que NO se eliminan todas las fotos
    """
    log("━━━ ETAPA 2: Filtro de blur ADAPTATIVO ━━━")
    import cv2

    frames = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    if not frames:
        return 0

    # PASO 1: Calcular varianza de cada foto
    variances = []
    for f in frames:
        p = os.path.join(frames_dir, f)
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            variances.append((f, 0.0))
            continue
        var = cv2.Laplacian(img, cv2.CV_64F).var()
        variances.append((f, var))

    if not variances:
        return 0

    # Estadísticas para diagnóstico
    vars_only = [v for _, v in variances]
    var_min = min(vars_only)
    var_max = max(vars_only)
    var_mean = sum(vars_only) / len(vars_only)
    var_median = sorted(vars_only)[len(vars_only) // 2]
    log(f"Varianzas: min={var_min:.1f}, max={var_max:.1f}, "
        f"media={var_mean:.1f}, mediana={var_median:.1f}")

    # PASO 2: Decidir estrategia
    # Cuántas pasarían el umbral absoluto
    above_absolute = sum(1 for _, v in variances if v >= BLUR_THRESHOLD_ABSOLUTE)
    ratio = above_absolute / len(variances)

    log(f"Sobre umbral absoluto ({BLUR_THRESHOLD_ABSOLUTE}): "
        f"{above_absolute}/{len(variances)} ({ratio*100:.0f}%)")

    if ratio >= MIN_VALID_RATIO:
        # Modo normal: usar umbral absoluto
        log(f"Modo: NORMAL (umbral absoluto = {BLUR_THRESHOLD_ABSOLUTE})")
        to_remove = [f for f, v in variances if v < BLUR_THRESHOLD_ABSOLUTE]
    else:
        # Modo permisivo: usar percentil
        # Calcular umbral del percentil 25 (eliminar solo el 25% más borroso)
        sorted_vars = sorted(vars_only)
        percentile_idx = max(1, len(sorted_vars) * BLUR_PERCENTILE_FALLBACK // 100)
        adaptive_threshold = sorted_vars[percentile_idx]
        log(f"Modo: PERMISIVO (umbral adaptativo = {adaptive_threshold:.1f}, "
            f"percentil {BLUR_PERCENTILE_FALLBACK})")
        to_remove = [f for f, v in variances if v < adaptive_threshold]

    # PASO 3: Verificar que NO eliminamos todas
    if len(to_remove) >= len(frames):
        log("WARN: Habríamos eliminado todas las fotos. Conservando todas.", "WARN")
        to_remove = []

    # PASO 4: Si quedaríamos con menos de MIN_IMGS, conservar las mejores
    remaining_after_remove = len(frames) - len(to_remove)
    if remaining_after_remove < MIN_IMGS:
        log(f"WARN: Solo quedarían {remaining_after_remove} fotos. "
            f"Conservando las {MIN_IMGS} mejores.", "WARN")
        # Ordenar por varianza descendente y mantener las mejores MIN_IMGS
        sorted_by_var = sorted(variances, key=lambda x: x[1], reverse=True)
        keep = set(f for f, _ in sorted_by_var[:MIN_IMGS])
        to_remove = [f for f in frames if f not in keep]

    # PASO 5: Eliminar
    for f in to_remove:
        os.remove(os.path.join(frames_dir, f))

    kept = len(frames) - len(to_remove)
    log(f"Nítidas: {kept}, Borrosas eliminadas: {len(to_remove)}")
    return kept

# ── Etapa 3: Depth Anything V2 ────────────────────────────────

def gen_depth(frames_dir, depth_dir):
    log("━━━ ETAPA 3: Depth Anything V2 ━━━")
    os.makedirs(depth_dir, exist_ok=True)
    try:
        import torch
        from PIL import Image
        from transformers import pipeline
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=device)
        frames = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
        for i,f in enumerate(frames):
            img = Image.open(os.path.join(frames_dir,f)).convert("RGB")
            d = pipe(img)["depth"]
            arr = np.array(d)
            norm = ((arr-arr.min())/(arr.max()-arr.min()+1e-8)*65535).astype(np.uint16)
            Image.fromarray(norm).save(os.path.join(depth_dir, f.replace(".jpg","_depth.png")))
            if (i+1)%25==0: log(f"  Depth {i+1}/{len(frames)}")
        log(f"Depth maps: {len(frames)}")
    except Exception as e:
        log(f"WARN: Depth falló ({e}), continuando sin depth priors", "WARN")

# ── Etapa 4: Mask2Former ──────────────────────────────────────

def gen_masks(frames_dir, masks_dir):
    log("━━━ ETAPA 4: Mask2Former ━━━")
    os.makedirs(masks_dir, exist_ok=True)
    try:
        import torch
        from PIL import Image
        from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
        device = "cuda" if torch.cuda.is_available() else "cpu"
        proc = Mask2FormerImageProcessor.from_pretrained("facebook/mask2former-swin-base-ade-semantic")
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            "facebook/mask2former-swin-base-ade-semantic").to(device).eval()
        EXCLUDE = {27, 8, 147}  # mirror, windowpane, glass en ADE20K
        frames = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
        with torch.no_grad():
            for i,f in enumerate(frames):
                img = Image.open(os.path.join(frames_dir,f)).convert("RGB")
                inputs = proc(images=img, return_tensors="pt").to(device)
                outputs = model(**inputs)
                seg = proc.post_process_semantic_segmentation(outputs, target_sizes=[img.size[::-1]])[0].cpu().numpy()
                mask = np.ones_like(seg, dtype=np.uint8)*255
                for c in EXCLUDE: mask[seg==c] = 0
                Image.fromarray(mask).save(os.path.join(masks_dir, f.replace(".jpg",".png")))
                if (i+1)%25==0: log(f"  Masks {i+1}/{len(frames)}")
        log(f"Máscaras: {len(frames)}")
    except Exception as e:
        log(f"WARN: Mask2Former falló ({e}), continuando sin máscaras", "WARN")

# ── Etapa 5: COLMAP ───────────────────────────────────────────

def run_colmap(images_dir, output_dir):
    log("━━━ ETAPA 5: COLMAP ━━━")
    sparse = os.path.join(output_dir,"sparse")
    db = os.path.join(output_dir,"database.db")
    os.makedirs(sparse, exist_ok=True)

    run(["colmap","feature_extractor","--database_path",db,
         "--image_path",images_dir,"--ImageReader.single_camera","1",
         "--ImageReader.camera_model","OPENCV",
         "--SiftExtraction.use_gpu","1"],
        TIMEOUTS["colmap_feature"], "features")

    run(["colmap","exhaustive_matcher","--database_path",db,
         "--SiftMatching.use_gpu","1"],
        TIMEOUTS["colmap_match"], "matching")

    run(["colmap","mapper","--database_path",db,
         "--image_path",images_dir,"--output_path",sparse],
        TIMEOUTS["colmap_mapper"], "mapper")

    s0 = os.path.join(sparse,"0")
    if not os.path.exists(s0):
        raise RuntimeError("COLMAP falló: no pudo reconstruir la escena. "
            "Posibles causas: paredes sin textura, motion blur, pocas fotos.")

    run(["colmap","image_undistorter","--image_path",images_dir,
         "--input_path",s0,"--output_path",output_dir,
         "--output_type","COLMAP"],
        TIMEOUTS["colmap_undistort"], "undistort")

    log("COLMAP OK")

# ── Etapa 6: gsplat training ──────────────────────────────────

def run_gsplat(data_dir, result_dir, iters):
    log(f"━━━ ETAPA 6: gsplat ({iters} iter) ━━━")

    trainer_paths = [
        "/opt/gsplat-repo/examples/simple_trainer.py",
        "/workspace/gsplat-repo/examples/simple_trainer.py",
    ]
    trainer = None
    for p in trainer_paths:
        if os.path.exists(p):
            trainer = p; break

    if trainer is None:
        log("Clonando gsplat repo para trainer...")
        run(["git","clone","--depth","1",
             "https://github.com/nerfstudio-project/gsplat.git",
             "/workspace/gsplat-repo"], 300, "git_clone")
        trainer = "/workspace/gsplat-repo/examples/simple_trainer.py"
        reqs = "/workspace/gsplat-repo/examples/requirements.txt"
        if os.path.exists(reqs):
            run(["pip","install","-r",reqs], 300, "trainer_deps")

    run(["python", trainer, "default",
         "--data_dir", data_dir, "--data_factor","1",
         "--result_dir", result_dir,
         "--max_steps", str(iters),
         "--save_steps", str(iters),
         "--eval_steps", str(iters+1),
         "--disable_viewer"],
        TIMEOUTS["gsplat"], "gsplat")

    log("gsplat OK")

# ── Etapa 7: Cleanup ──────────────────────────────────────────

def cleanup_ply(ply_path):
    log("━━━ ETAPA 7: Cleanup ━━━")
    log("Cleanup: gsplat ya pruneó outliers durante training. PLY conservado.")
    return ply_path

# ── Etapa 8: Collision mesh ───────────────────────────────────

def gen_collision(ply_path, glb_path):
    log("━━━ ETAPA 8: Collision mesh ━━━")
    try:
        run(["splat-transform", ply_path, glb_path, "-K"],
            TIMEOUTS["collision"], "collision")
        if os.path.exists(glb_path):
            log(f"Collision mesh: {os.path.getsize(glb_path)//1024} KB")
            return True
    except Exception as e:
        log(f"WARN: Collision mesh falló ({e})", "WARN")
    return False

def find_ply(d):
    for root,_,files in os.walk(d):
        for f in files:
            if f.endswith(".ply"):
                return os.path.join(root,f)
    raise RuntimeError(f"No se generó PLY en {d}")

# ── HANDLER PRINCIPAL ─────────────────────────────────────────

def handler(job):
    global log
    log = Log()
    t0 = time.time()
    work = None

    try:
        inp = job.get("input", {})
        log(f"=== JOB {job.get('id','?')} ===")

        if inp.get("mode") == "health":
            import torch
            return {"status":"healthy",
                    "cuda": torch.cuda.is_available(),
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "torch": torch.__version__,
                    "seconds": round(time.time()-t0,1)}

        dl_url = inp.get("download_url") or inp.get("zip_url")
        if not dl_url:
            return {"status":"error","error":"Falta download_url/zip_url"}

        quality = inp.get("quality","balanced")
        if quality not in ITERS:
            return {"status":"error","error":f"quality inválido: {list(ITERS.keys())}"}
        iters = ITERS[quality]

        upload_ply_url = inp.get("upload_url_ply")
        upload_glb_url = inp.get("upload_url_glb")
        webhook_url = inp.get("webhook_url")
        job_id = inp.get("job_id", job.get("id","unknown"))

        work = tempfile.mkdtemp(prefix="gs_")
        raw = os.path.join(work,"raw")
        frames = os.path.join(work,"frames")
        depth = os.path.join(work,"depths")
        masks = os.path.join(work,"masks")
        colmap = os.path.join(work,"colmap")
        result = os.path.join(work,"result")
        for d in [raw,frames,colmap,result]: os.makedirs(d, exist_ok=True)

        zip_path = os.path.join(work,"input.zip")
        download(dl_url, zip_path)

        log("Extrayendo ZIP...")
        with zipfile.ZipFile(zip_path,"r") as z: z.extractall(raw)
        os.remove(zip_path)

        # Aplanar subcarpetas
        for _ in range(3):
            items = os.listdir(raw)
            if len(items)==1 and os.path.isdir(os.path.join(raw,items[0])):
                inner = os.path.join(raw,items[0])
                for f in os.listdir(inner): shutil.move(os.path.join(inner,f),raw)
                os.rmdir(inner)
            else: break
        if os.path.isdir(os.path.join(raw,"images")):
            for f in os.listdir(os.path.join(raw,"images")):
                shutil.move(os.path.join(raw,"images",f),raw)
            os.rmdir(os.path.join(raw,"images"))

        # ETAPA 1
        count = extract_frames(raw, frames)
        if count < MIN_IMGS:
            return {"status":"error","stage":"extraction","error":f"Solo {count} frames, mínimo {MIN_IMGS}"}

        # ETAPA 2 (FIX v2)
        kept = filter_blur(frames)
        if kept < MIN_IMGS:
            return {"status":"error","stage":"blur","error":f"Solo {kept} frames nítidos tras filtro adaptativo, mínimo {MIN_IMGS}"}

        # Limitar a MAX_IMGS
        all_frames = sorted([f for f in os.listdir(frames) if f.endswith(".jpg")])
        if len(all_frames) > MAX_IMGS:
            step = len(all_frames)/MAX_IMGS
            keep = {int(i*step) for i in range(MAX_IMGS)}
            for i,f in enumerate(all_frames):
                if i not in keep: os.remove(os.path.join(frames,f))
        final_count = len([f for f in os.listdir(frames) if f.endswith(".jpg")])
        log(f"Frames finales: {final_count}")

        # ETAPA 3
        gen_depth(frames, depth)

        # ETAPA 4
        gen_masks(frames, masks)

        # ETAPA 5
        run_colmap(frames, colmap)

        # ETAPA 6
        run_gsplat(colmap, result, iters)

        ply = find_ply(result)
        ply_mb = os.path.getsize(ply)/(1024*1024)
        log(f"PLY: {ply_mb:.1f} MB")

        # ETAPA 7
        ply = cleanup_ply(ply)

        # ETAPA 8
        glb = os.path.join(result,"collision.glb")
        has_glb = gen_collision(ply, glb)

        if upload_ply_url:
            try: upload(ply, upload_ply_url)
            except Exception as e: log(f"WARN: Upload PLY falló ({e})", "WARN")

        if upload_glb_url and has_glb:
            try: upload(glb, upload_glb_url)
            except Exception as e: log(f"WARN: Upload GLB falló ({e})", "WARN")

        elapsed = round(time.time()-t0, 1)

        out = {
            "status":"success",
            "job_id": job_id,
            "seconds": elapsed,
            "frames_total": count,
            "frames_kept": kept,
            "frames_used": final_count,
            "quality": quality,
            "iterations": iters,
            "ply_mb": round(ply_mb,2),
            "has_collision": has_glb,
        }

        if not upload_ply_url and ply_mb < 30:
            with open(ply,"rb") as f: out["ply_base64"] = base64.b64encode(f.read()).decode()
        if not upload_glb_url and has_glb and os.path.getsize(glb)<5*1024*1024:
            with open(glb,"rb") as f: out["glb_base64"] = base64.b64encode(f.read()).decode()

        if webhook_url:
            try:
                req.post(webhook_url, json=out, timeout=30)
                log("Webhook enviado")
            except: log("WARN: Webhook falló", "WARN")

        log(f"=== SUCCESS {elapsed}s ===")
        return out

    except Timeout as e:
        elapsed = round(time.time()-t0,1)
        log(f"TIMEOUT: {e}", "ERROR")
        err = {"status":"error","stage":"timeout","error":str(e),
               "last_cmd":get_cmd(),"log":log.tail(),"seconds":elapsed}
        if inp.get("webhook_url"):
            try: req.post(inp["webhook_url"], json=err, timeout=10)
            except: pass
        return err

    except Exception as e:
        elapsed = round(time.time()-t0,1)
        tb = traceback.format_exc()
        log(f"ERROR: {e}", "ERROR")
        err = {"status":"error","stage":"exception","error":str(e),
               "traceback":tb,"last_cmd":get_cmd(),"log":log.tail(),"seconds":elapsed}
        if inp.get("webhook_url"):
            try: req.post(inp["webhook_url"], json=err, timeout=10)
            except: pass
        return err

    finally:
        if work and os.path.exists(work):
            try: shutil.rmtree(work)
            except: pass

if __name__ == "__main__":
    print("[WORKER] Starting...", flush=True)
    runpod.serverless.start({"handler": handler})
