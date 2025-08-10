# app.py
import os
import re
import uuid
import shutil
import subprocess
import numpy as np
import cv2 as cv

from threading import Thread
from datetime import datetime, date
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# --- Dynamsoft Barcode Reader ---
import dbr
from dbr import BarcodeReader, BarcodeReaderError

# --- EasyOCR para fecha de vencimiento ---
import easyocr

# --- SQLAlchemy (PostgreSQL) ---
from sqlalchemy import (
    create_engine, MetaData, Table, Column,
    String, DateTime, Date, BigInteger, select, update, delete, UniqueConstraint, nulls_last
)
from sqlalchemy.exc import SQLAlchemyError

# --- PyTorch para Abierto/Cerrado ---
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

# =========================
# Configuración de la App
# =========================
app = FastAPI(title="UNMSM - Web Scanner DBR + Fecha + Open/Closed")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# Carpeta estática
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# =========================
# DBR Init
# =========================
dbr_license = os.getenv("DBR_LICENSE", "").strip()
if not dbr_license:
    dbr_license = (
        "t0081YQEAABpA4FtLtYevgUwlxMA5EHD70udzTDfr/IndkBn2n4sqcxSgeEvqiuhz8ZMdubukK4BrxHVmbPi4vn5nxnIirybfNN43i8v2TA5+oEeq;"
        "t0083YQEAAJ8jGUQXXSp5kkavrIw/bC74Gy0x1LsNtHoSWn7RdRYJjxblCXiNdU1uWUmyc3lwv9RLtYMXUton0tCu29HVUkv6903D+2YTba3iDhTISFc=;"
        "t0083YQEAALjfywnsZjEKDkecIcIJB7ynMlpuQGD/1TmoQbCAU+88tCDWOf4IQZgH3GioE7m9nK28LIOUGr03BB3Onjr5gcqM753Ge2dy2V6rThFMSFY="
    )
dbr.BarcodeReader.init_license(dbr_license)
reader = BarcodeReader()

# =========================
# EasyOCR Init
# =========================
ocr_reader = easyocr.Reader(['es', 'en'])

PALABRAS_CLAVE = [
    'vence', 'vencimiento', 'vto', 'exp', 'expira',
    'cad', 'caduca', 'valido', 'validez', 'usese antes de', 'best before',
    'use by', 'use before', 'expiry', 'expiration'
]

def contiene_contexto(texto: str):
    for palabra in PALABRAS_CLAVE:
        if re.search(rf'\b{re.escape(palabra)}\b', texto, re.IGNORECASE):
            return palabra
    return None

def detectar_fecha_en_texto(texto: str):
    texto = re.sub(r'(?<=\d)[\s]+(?=[/-])', '', texto)
    texto = re.sub(r'(?<=[/-])[\s]+(?=\d)', '', texto)
    texto = re.sub(r'\s{2,}', ' ', texto)

    patrones = [
        r'\b\d{2}[/-]\d{2}[/-]\d{4}\b',
        r'\b\d{2}[/-]\d{2}[/-]\d{2}\b',
        r'\b\d{4}[/-]\d{2}[/-]\d{2}\b',
        r'\b\d{4}[/-]\d{2}[/-]\d{2}[ T]\d{2}:\d{2}:\d{2}\b',
        r'\b\d{4}[/-]\d{2}[/-]\d{2}[ T]\d{2}:\d{2}\b',
        r'\b\d{2} \w{3,} \d{4}\b',
        r'\b\w{3,} \d{2}, \d{4}\b',
        r'\b\w{3,} \d{4}\b',
        r'\b\d{2}[/-]\d{4}\b',
        r'\b\d{4}[/-]\d{2}\b',
        r'\b\d{1,2} [a-zA-Z]{3,9} \d{4} - \d{1,2} [a-zA-Z]{3,9} \d{4}\b',
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4} ?- ?\d{1,2}[/-]\d{1,2}[/-]\d{4}\b',
        r'\b\w{3,9} \d{4} ?– ?\w{3,9} \d{4}\b',
        r'\b\d{2}[A-Z]{3}\d{2}\b',
        r'\b\d{2}:\d{2}\b',
        r'\b\d{2}:\d{2}:\d{2}\b'
    ]
    for pat in patrones:
        m = re.search(pat, texto, re.IGNORECASE)
        if m:
            return m.group()
    return None

def es_fecha_valida(fecha_str: str):
    if not fecha_str:
        return None
    formatos = [
        "%d/%m/%Y", "%d/%m/%y",
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M",
        "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"
    ]
    for fmt in formatos:
        try:
            return datetime.strptime(fecha_str, fmt)
        except ValueError:
            continue
    return None

# =========================
# Persistencia en PostgreSQL
# =========================
POSTGRES_DSN = os.getenv(
    "POSTGRES_DSN",
    "postgresql+psycopg2://postgres:lvame8d@localhost:5432/postgres"
)
engine = create_engine(POSTGRES_DSN, pool_pre_ping=True, future=True)
metadata = MetaData()
productos = Table(
    "productos", metadata,
    Column("tipo", String(64), nullable=False),
    Column("codigo", BigInteger, nullable=False),
    Column("timestamp", DateTime, nullable=False),
    Column("fecha_vencimiento", Date, nullable=True),
    UniqueConstraint("codigo", name="uq_productos_codigo")
)
with engine.begin() as conn:
    metadata.create_all(conn)

codigos_detectados_sesion = set()

def guardar_en_db(texto: str, tipo: str):
    if texto in codigos_detectados_sesion:
        return False, "Ya registrado en esta sesión"
    try:
        codigo_entero = int(texto)
    except ValueError:
        return False, "Código no válido como entero"

    try:
        with engine.begin() as conn:
            exists = conn.execute(
                select(productos.c.codigo).where(productos.c.codigo == codigo_entero)
            ).first()
            if exists is None:
                conn.execute(productos.insert().values(
                    tipo=tipo, codigo=codigo_entero, timestamp=datetime.now(), fecha_vencimiento=None
                ))
                codigos_detectados_sesion.add(texto)
                return True, "Registrado correctamente"
            else:
                codigos_detectados_sesion.add(texto)
                return False, "Ya existía en la base"
    except SQLAlchemyError as e:
        return False, f"Error de base: {str(e)}"

def guardar_fecha_vencimiento_en_ultima_fila(fecha_iso: str):
    if not fecha_iso:
        return False, "No hay fecha que guardar"
    try:
        y, m, d = map(int, fecha_iso.split("-"))
        fecha_dt = date(y, m, d)
    except Exception:
        return False, "Formato de fecha inválido (YYYY-MM-DD)"

    try:
        with engine.begin() as conn:
            max_ts = conn.execute(
                select(productos.c.timestamp).order_by(productos.c.timestamp.desc()).limit(1)
            ).scalar_one_or_none()
            if max_ts is None:
                return False, "No hay filas para actualizar (primero escanea un código)."
            conn.execute(
                update(productos).where(productos.c.timestamp == max_ts).values(fecha_vencimiento=fecha_dt)
            )
            return True, f"Fecha guardada en el último registro: {fecha_iso}"
    except SQLAlchemyError as e:
        return False, f"Error de base: {str(e)}"

def limpiar_db():
    try:
        with engine.begin() as conn:
            conn.execute(delete(productos))
        return True, "Tabla 'productos' limpiada (contenido borrado)."
    except SQLAlchemyError as e:
        return False, f"Error limpiando la tabla: {str(e)}"

# =========================
# Decodificación (con ROI) - CÓDIGOS
# =========================
def decode_image_bytes_with_roi(image_bytes: bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv.imdecode(np_arr, cv.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        return []

    h, w = frame.shape[:2]
    rw = min(400, w); rh = min(200, h)
    x1 = (w - rw)//2; y1 = (h - rh)//2
    roi = frame[y1:y1+rh, x1:x1+rw]

    try:
        results = reader.decode_buffer(roi)
    except BarcodeReaderError as e:
        print("[DBR] Error:", e, flush=True)
        results = None

    parsed = []
    if results:
        for r in results:
            texto = r.barcode_text
            fmt = r.barcode_format_string
            if texto.startswith("Attention("):
                continue
            saved, msg = guardar_en_db(texto, fmt)

            pts_abs = []
            if r.localization_result and r.localization_result.localization_points:
                for p in r.localization_result.localization_points:
                    px = max(0, min(int(p[0]) + x1, w - 1))
                    py = max(0, min(int(p[1]) + y1, h - 1))
                    pts_abs.append([px, py])

            parsed.append({
                "text": texto, "format": fmt, "points": pts_abs,
                "saved": saved, "save_message": msg
            })
    return parsed

# =========================
# OCR Fecha (mismo ROI)
# =========================
def detect_expiry_in_image(image_bytes: bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv.imdecode(np_arr, cv.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        return {"raw_text": "", "context": None, "matched_date": None, "valid_date_iso": None, "roi": None}

    h, w = frame.shape[:2]
    rw = min(400, w); rh = min(200, h)
    x1 = (w - rw)//2; y1 = (h - rh)//2

    roi = frame[y1:y1+rh, x1:x1+rw]
    roi_big = cv.resize(roi, None, fx=2.5, fy=2.5, interpolation=cv.INTER_LINEAR)

    ocr_texts = ocr_reader.readtext(roi_big, detail=0)
    raw_text = " ".join(ocr_texts).strip()

    if not raw_text:
        return {"raw_text": "", "context": None, "matched_date": None, "valid_date_iso": None, "roi": [x1, y1, x1+rw, y1+rh]}

    ctx = contiene_contexto(raw_text)
    matched = detectar_fecha_en_texto(raw_text)
    valid_dt = es_fecha_valida(matched) if matched else None
    valid_iso = valid_dt.strftime("%Y-%m-%d") if valid_dt else None

    return {
        "raw_text": raw_text, "context": ctx, "matched_date": matched,
        "valid_date_iso": valid_iso, "roi": [x1, y1, x1+rw, y1+rh]
    }

# =========================
# Modelo Abierto/Cerrado (PyTorch)
# =========================
CLASS_NAMES = ["abierto", "cerrado"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CROP_LEFT  = int(os.getenv("OPENCLS_CROP_LEFT", "440"))
CROP_RIGHT = int(os.getenv("OPENCLS_CROP_RIGHT", "440"))
THRESH_ABIERTO = float(os.getenv("OPENCLS_THRESH", "0.95"))

MODEL_PATH = os.getenv(
    "OPENCLS_MODEL_PATH",
    "C:/Users/yvonn/OneDrive/Documentos/Proyectos/OpenProducts/models/resnet18_all_products.pth"
)

_opencls_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

_opencls_model = models.resnet18(weights=None)
_opencls_model.fc = torch.nn.Linear(_opencls_model.fc.in_features, 2)

_model_loaded = False
_model_error = None
try:
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    _opencls_model.load_state_dict(state)
    _opencls_model.to(DEVICE)
    _opencls_model.eval()
    _model_loaded = True
except Exception as e:
    _model_error = str(e)
    print(f"[Open/Closed] No se pudo cargar el modelo: {e}", flush=True)

def predict_open_closed(frame_bgr):
    """Devuelve (label, confidence, cropped_bgr)."""
    h, w, _ = frame_bgr.shape
    left = max(0, min(CROP_LEFT, w-1))
    right = max(0, min(CROP_RIGHT, w-1))
    cropped = frame_bgr if (left + right >= w) else frame_bgr[:, left:w-right]

    if not _model_loaded:
        return "cerrado", 1.0, cropped

    img_pil = Image.fromarray(cv.cvtColor(cropped, cv.COLOR_BGR2RGB))
    input_tensor = _opencls_transform(img_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = _opencls_model(input_tensor)
        probs = F.softmax(output, dim=1)[0]
        abierto_prob = probs[0].item()
        cerrado_prob = probs[1].item()
        if abierto_prob > THRESH_ABIERTO:
            return "abierto", float(abierto_prob), cropped
        else:
            return "cerrado", float(cerrado_prob), cropped

# =========================
# Estado de jobs para progreso en vivo
# =========================
JOBS = {}  # job_id -> {"total":0, "done":0, "open":0, "closed":0, "ready":False, "result":None, "error":None}

def _process_video_job(job_id: str, in_path: str, stride: int):
    try:
        cap = cv.VideoCapture(in_path)
        if not cap.isOpened():
            JOBS[job_id]["error"] = "No se pudo abrir el video"
            print(f"[job {job_id[:6]}] ERROR: no se pudo abrir el video", flush=True)
            return

        frames_total = int(cap.get(cv.CAP_PROP_FRAME_COUNT)) or 0
        fps_est = float(cap.get(cv.CAP_PROP_FPS)) if cap.get(cv.CAP_PROP_FPS) else 25.0
        fps_out = max(1.0, fps_est / max(1, stride))
        JOBS[job_id]["total"] = frames_total

        print(f"[job {job_id[:6]}] Inicio · total={frames_total} · stride={stride} · fps_in={fps_est:.2f} · fps_out={fps_out:.2f}", flush=True)

        ts_tag = datetime.now().strftime('%Y%m%d_%H%M%S') + "_" + job_id[:6]
        tmp_avi = os.path.join("static", f"_annot_{ts_tag}.avi")
        out_mp4 = os.path.join("static", f"annotated_{ts_tag}.mp4")
        out_webm = os.path.join("static", f"annotated_{ts_tag}.webm")
        fourcc = cv.VideoWriter_fourcc(*'MJPG')
        writer = None

        i = 0
        frames_proc = 0
        timeline = []
        segments = []
        run_label = None
        run_start = None
        probs = []
        last_log = 0  # para throttling de logs

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if stride > 1 and (i % stride != 0):
                i += 1
                continue

            label, conf, cropped = predict_open_closed(frame)

            if writer is None:
                h, w = cropped.shape[:2]
                writer = cv.VideoWriter(tmp_avi, fourcc, fps_out, (w, h))

            # Overlay
            text = f"{label.upper()} ({conf:.2f})"
            color = (0, 255, 0) if label == "cerrado" else (0, 0, 255)
            cv.putText(cropped, text, (30, 40), cv.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv.LINE_AA)
            cv.rectangle(cropped, (20, 15), (30, 25), color, -1)
            writer.write(cropped)

            # progreso
            JOBS[job_id]["done"] += 1
            if label == "abierto":
                JOBS[job_id]["open"] += 1
            else:
                JOBS[job_id]["closed"] += 1

            # timeline + segmentos
            timeline.append({"i_frame": i, "label": label, "prob": float(conf)})
            if run_label is None:
                run_label, run_start, probs = label, i, [conf]
            elif label == run_label:
                probs.append(conf)
            else:
                run_end = i - 1
                t0 = run_start / fps_est
                t1 = run_end / fps_est
                segments.append({
                    "label": run_label,
                    "start_frame": run_start,
                    "end_frame": run_end,
                    "start_time": round(t0, 2),
                    "end_time": round(t1, 2),
                    "duration_s": round(max(0, t1 - t0), 2),
                    "avg_prob": round(sum(probs) / max(1, len(probs)), 3)
                })
                run_label, run_start, probs = label, i, [conf]

            frames_proc += 1
            i += 1

            # Log periódico en consola del servidor
            if JOBS[job_id]["done"] - last_log >= 50 or JOBS[job_id]["done"] == frames_total:
                done = JOBS[job_id]["done"]
                abierto = JOBS[job_id]["open"]
                cerrado = JOBS[job_id]["closed"]
                pct = (done / max(1, frames_total)) * 100.0
                pct_open = (abierto / max(1, done)) * 100.0 if done else 0.0
                pct_closed = (cerrado / max(1, done)) * 100.0 if done else 0.0
                bar = "#" * int(pct/5)
                print(f"[job {job_id[:6]}] {done}/{frames_total} [{bar:<20}] {pct:5.1f}% · ABIERTO {pct_open:5.2f}% · CERRADO {pct_closed:5.2f}%", flush=True)
                last_log = done

        if run_label is not None:
            run_end = timeline[-1]["i_frame"] if timeline else 0
            t0 = run_start / fps_est
            t1 = run_end / fps_est
            segments.append({
                "label": run_label,
                "start_frame": run_start,
                "end_frame": run_end,
                "start_time": round(t0, 2),
                "end_time": round(t1, 2),
                "duration_s": round(max(0, t1 - t0), 2),
                "avg_prob": round(sum(probs) / max(1, len(probs)), 3)
            })

        cap.release()
        if writer is not None:
            writer.release()

        # Transcodificar si hay ffmpeg
        annotated_url_mp4 = None
        annotated_url_webm = None
        if shutil.which("ffmpeg") and os.path.exists(tmp_avi):
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-i", tmp_avi,
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                    "-movflags", "+faststart", out_mp4
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if os.path.exists(out_mp4):
                    annotated_url_mp4 = f"/static/{os.path.basename(out_mp4)}"
            except Exception:
                pass
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-i", tmp_avi,
                    "-c:v", "libvpx-vp9", "-b:v", "1M", out_webm
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if os.path.exists(out_webm):
                    annotated_url_webm = f"/static/{os.path.basename(out_webm)}"
            except Exception:
                pass

        if not annotated_url_mp4 and os.path.exists(tmp_avi):
            annotated_url_mp4 = f"/static/{os.path.basename(tmp_avi)}"

        JOBS[job_id]["ready"] = True
        JOBS[job_id]["result"] = {
            "frames_total": frames_total,
            "frames_procesados": frames_proc,
            "fps_est": fps_est,
            "fps_out": fps_out,
            "stride": stride,
            "timeline": timeline,
            "segments": segments,
            "annotated_url_mp4": annotated_url_mp4,
            "annotated_url_webm": annotated_url_webm,
            "ffmpeg": shutil.which("ffmpeg") is not None,
            "model_loaded": _model_loaded,
            "model_error": _model_error
        }
        print(f"[job {job_id[:6]}] FIN ✔️", flush=True)
    except Exception as e:
        JOBS[job_id]["error"] = str(e)
        print(f"[job {job_id[:6]}] ERROR: {e}", flush=True)
    finally:
        try:
            os.path.exists(in_path) and os.remove(in_path)
        except Exception:
            pass

# =========================
# HTML + Frontend (modo por defecto: CÓDIGOS)
# =========================
INDEX_HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Web Scanner (Códigos ↔ Fecha ↔ Open/Closed)</title>
  <script src="https://cdn.jsdelivr.net/npm/sweetalert2@11"></script>
  <style>
    :root { --bg:#111; --fg:#eee; --muted:#999; --card:#171717; --border:#2b2b2b; }
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; padding: 0; background:var(--bg); color:var(--fg); }
    .wrap { display:grid; gap:12px; padding:16px; width:100%; max-width:1200px; margin:auto; }
    .hud { display:flex; gap:10px; align-items:center; flex-wrap: wrap; }
    button { padding:8px 14px; border-radius:10px; background:#222; color:#eee; border:1px solid #444; cursor:pointer; }
    button:disabled { opacity:.5; cursor:not-allowed; }
    .badge { padding:6px 10px; border-radius:10px; background:#222; border:1px solid #444; }
    .ok { background:#1b5; border-color:#0a3; }
    .err { background:#b31; border-color:#811; }
    .main { display:grid; grid-template-columns: 1.2fr .8fr; gap:14px; }
    @media(max-width: 980px){ .main { grid-template-columns: 1fr; } }
    #layer { position: relative; width: 100%; }
    #layer > video, #layer > canvas { position: absolute; top:0; left:0; width:100%; height:auto; border-radius:12px; display:block; }
    #overlay { z-index:2; pointer-events:none; }
    video { z-index:1; background:#000; }
    #results, #expiry { position:relative; z-index:0; }
    #results, #expiry { font-size:14px; display:grid; gap:6px; }
    .card { border-radius:10px; padding:10px 12px; border:1px solid var(--border); background:var(--card); }
    .card.barcode { border-color:#1f7a1f; box-shadow:0 0 0 1px rgba(31,122,31,.15) inset; }
    .card.date    { border-color:#1e6fbf; box-shadow:0 0 0 1px rgba(30,111,191,.15) inset; }
    .panel { border:1px solid var(--border); border-radius:12px; padding:10px; background:#141414; }
    .panel h3 { margin:6px 0 10px; font-weight:600; }
    table { width:100%; border-collapse: collapse; font-size: 13px; }
    th, td { padding:8px 10px; border-bottom:1px solid #2a2a2a; text-align:left; }
    th { color:#ddd; position:sticky; top:0; background:#151515; }
    .muted { color: var(--muted); }
    .tag { padding:3px 8px; border-radius:999px; font-size:12px; border:1px solid #333; display:inline-block; }
    .red { background:#3a0e0e; border-color:#7a1b1b; color:#ffb3b3; }
    .yellow { background:#3a360e; border-color:#7a6f1b; color:#ffef9a; }
    .green { background:#123a12; border-color:#1c7a1b; color:#b9f6b9; }
    .scroll { max-height: 520px; overflow:auto; border-radius:8px; }
    .swal2-popup { background: var(--card) !important; color: var(--fg) !important; border:1px solid var(--border); border-radius:12px; }
    .detector { margin-top:12px; padding-top:10px; border-top:1px dashed var(--border); }
    .detector h4 { margin: 0 0 8px 0; font-weight:600; }
    .row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
    .row input[type="number"] { width:90px; }
    video.preview { width: 100%; margin-top: 10px; border-radius: 10px; border:1px solid var(--border); background:#000; }
    a.btn { display:inline-block; padding:6px 12px; border-radius:10px; border:1px solid #2b2b2b; text-decoration:none; color:#eee; background:#202020; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hud">
      <button id="btnStart">Start</button>
      <button id="btnStop" disabled>Stop</button>
      <button id="btnClear">Limpiar Tabla</button>
      <button id="btnMode">Detectar fecha</button>
      <span class="badge" id="status">Idle</span>
      <span class="badge" id="last">Último: —</span>
    </div>

    <div class="main">
      <div>
        <div id="layer">
          <video id="video" autoplay playsinline muted></video>
          <canvas id="canvas" style="display:none;"></canvas>
          <canvas id="overlay"></canvas>
        </div>
        <div id="results" style="margin-top:10px;"></div>
        <div id="expiry"></div>

        <!-- Progreso en vivo -->
        <div id="liveStats" class="card" style="display:none; margin-top:10px;">
          <b>Modelo (en curso)</b><br/>
          ABIERTO: <span id="openPct">0%</span> &middot;
          CERRADO: <span id="closedPct">0%</span><br/>
          <small class="muted">Frames: <span id="done">0</span>/<span id="total">0</span></small>
        </div>
      </div>

      <aside class="panel">
        <h3>Productos (por vencimiento ↑)</h3>
        <div class="scroll">
          <table id="tbl">
            <thead>
              <tr>
                <th>Código</th>
                <th>Tipo</th>
                <th>Escaneado</th>
                <th>Vence</th>
                <th>Estado</th>
              </tr>
            </thead>
            <tbody id="tblBody"></tbody>
          </table>
        </div>
        <small class="muted">* Las filas sin fecha de vencimiento aparecen al final.</small>

        <div class="detector">
          <h4>Detección Abierto/Cerrado (desde video)</h4>
          <div class="row">
            <input type="file" id="vidFile" accept="video/*">
            <label for="stride" class="muted">stride</label>
            <input type="number" id="stride" min="1" value="10" title="Procesar 1 de cada N frames">
            <button id="btnDetectVideo" disabled>Analizar video</button>
          </div>
          <small class="muted">Procesa 1 de cada N frames (stride). Usa 10 para videos largos.</small>

          <div id="detSummary" class="card" style="margin-top:10px; display:none;"></div>

          <div id="detMedia" style="display:none;">
            <video id="detPreview" class="preview" controls>
              Tu navegador no soporta la reproducción de video.
            </video>
            <div style="margin-top:8px; display:flex; gap:8px; flex-wrap:wrap;">
              <a id="detDownloadMp4" class="btn" href="#" download>Descargar MP4</a>
              <a id="detDownloadWebm" class="btn" href="#" download>Descargar WebM</a>
            </div>
          </div>

          <div id="detSegments" class="card" style="margin-top:10px; display:none;">
            <b>Segmentos detectados</b>
            <div id="segTable" style="margin-top:6px;"></div>
            <small class="muted">Tip: haz clic en “Ver” para saltar al tramo del video.</small>
          </div>
        </div>
      </aside>
    </div>
  </div>

<script>
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const overlay = document.getElementById('overlay');
const layer  = document.getElementById('layer');
const ctxO = overlay.getContext('2d');
const statusEl = document.getElementById('status');
const lastEl = document.getElementById('last');
const resultsEl = document.getElementById('results');
const expiryEl = document.getElementById('expiry');
const btnStart = document.getElementById('btnStart');
const btnStop  = document.getElementById('btnStop');
const btnClear = document.getElementById('btnClear');
const btnMode  = document.getElementById('btnMode');
const tblBody  = document.getElementById('tblBody');

const vidFile  = document.getElementById('vidFile');
const btnDetectVideo = document.getElementById('btnDetectVideo');
const strideEl = document.getElementById('stride');
const detSummary = document.getElementById('detSummary');
const detMedia = document.getElementById('detMedia');
const detPreview = document.getElementById('detPreview');
const detDownloadMp4 = document.getElementById('detDownloadMp4');
const detDownloadWebm = document.getElementById('detDownloadWebm');
const detSegments = document.getElementById('detSegments');
const segTable = document.getElementById('segTable');

/* Live progress */
const liveStats = document.getElementById('liveStats');
const openPctEl = document.getElementById('openPct');
const closedPctEl = document.getElementById('closedPct');
const doneEl = document.getElementById('done');
const totalEl = document.getElementById('total');

let stream = null;
let timer  = null;
let mode   = 'barcode';
let poller = null;
let progTimer = null;

const swalToast = Swal.mixin({ toast:true, position:'top-end', showConfirmButton:false, timer:2200, timerProgressBar:true });
function notify(icon,title,text=''){ swalToast.fire({icon,title,text}); }
function modal(icon,title,text){ return Swal.fire({icon,title,text}); }

function fixLayerHeight(){ if (overlay && overlay.clientHeight) layer.style.height = overlay.clientHeight + 'px'; else if (video && video.clientHeight) layer.style.height = video.clientHeight + 'px'; }
function syncCanvasSizes(){ const w=video.videoWidth,h=video.videoHeight; if(!w||!h) return false; canvas.width=overlay.width=w; canvas.height=overlay.height=h; fixLayerHeight(); return true; }
function drawGuideROI(color='yellow', showMsg=false){ const w=overlay.width,h=overlay.height;if(!w||!h)return; const rw=Math.min(400,w),rh=Math.min(200,h); const x1=Math.round((w-rw)/2),y1=Math.round((h-rh)/2); const c=ctxO; c.save(); c.lineWidth=2; c.strokeStyle=color; c.setLineDash([8,6]); c.strokeRect(x1,y1,rw,rh); c.restore(); if(showMsg){ const msg="Presiona 'C' para detectar fecha"; c.save(); c.font='16px sans-serif'; c.fillStyle='white'; c.strokeStyle='black'; c.lineWidth=3; c.strokeText(msg,x1,Math.max(16,y1-8)); c.fillText(msg,x1,Math.max(16,y1-8)); c.restore(); } }
function drawOverlayBoxes(results){ ctxO.clearRect(0,0,overlay.width, overlay.height); drawGuideROI(mode==='barcode'?'yellow':'deepskyblue', mode==='expiry'); if(mode!=='barcode') return; results.forEach(r=>{ if(r.points&&r.points.length===4){ ctxO.beginPath(); ctxO.moveTo(r.points[0][0],r.points[0][1]); for(let i=1;i<4;i++) ctxO.lineTo(r.points[i][0], r.points[i][1]); ctxO.closePath(); ctxO.lineWidth=3; ctxO.strokeStyle='lime'; ctxO.stroke(); } }); }
function nowISOish(){ const d=new Date(); const pad=n=>String(n).padStart(2,'0'); return `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`; }

function setBarcodeMode(){
  mode = 'barcode';
  btnMode.textContent = 'Detectar fecha';
  statusEl.textContent = 'Modo códigos';
  statusEl.className = 'badge';
  drawOverlayBoxes([]);
  if (stream && !timer) timer = setInterval(captureAndSendBarcode, 300);
}
function setExpiryMode(){
  if (timer){ clearInterval(timer); timer=null; }
  mode = 'expiry';
  btnMode.textContent = 'Detectar códigos';
  statusEl.textContent = "Modo fecha — Presiona 'C' para leer";
  statusEl.className = 'badge';
  drawOverlayBoxes([]);
}

window.addEventListener('load', ()=>{ setBarcodeMode(); });

function renderBarcodeCard(last){
  if(!last) return;
  resultsEl.innerHTML = `
    <div class="card barcode">
      <b>Fecha actual:</b> ${nowISOish()}<br/>
      <b>Tipo:</b> Código de barras<br/>
      <b>Formato:</b> ${last.format||'—'}<br/>
      <b>Código:</b> ${last.text||'—'}<br/>
      <b>Guardado:</b> ${last.saved?'Sí':'No'} ${last.save_message?('— '+last.save_message):''}
    </div>`;
  lastEl.textContent = 'Último: ' + (last.text||'—');
  if (last.saved) notify('success','Código guardado', `${last.text}`);
}
function appendResults(results){ if(!results||results.length===0) return; renderBarcodeCard(results[0]); }

async function captureAndSendBarcode(){
  try{
    if(!syncCanvasSizes()||mode!=='barcode') return;
    const w=video.videoWidth,h=video.videoHeight;
    const ctx=canvas.getContext('2d');
    ctx.drawImage(video,0,0,w,h);
    const blob=await new Promise(res=>canvas.toBlob(res,'image/jpeg',0.8));
    const form=new FormData();
    form.append('image',blob,'frame.jpg');
    const resp=await fetch('/decode',{method:'POST',body:form});
    if(!resp.ok) throw new Error('HTTP '+resp.status);
    const data=await resp.json();
    drawOverlayBoxes(data.results||[]);
    appendResults(data.results||[]);
    statusEl.textContent='OK (códigos)';
    statusEl.className='badge ok';
  }catch(e){
    statusEl.textContent='Error';
    statusEl.className='badge err';
    console.error(e);
  }
}

async function captureExpiryOnce(){
  try{
    if(!syncCanvasSizes()) return;
    const w=video.videoWidth,h=video.videoHeight;
    const ctx=canvas.getContext('2d');
    ctx.drawImage(video,0,0,w,h);
    const blob=await new Promise(res=>canvas.toBlob(res,'image/jpeg',0.9));
    const form=new FormData();
    form.append('image',blob,'frame.jpg');
    const resp=await fetch('/expiry',{method:'POST',body:form});
    if(!resp.ok) throw new Error('HTTP '+resp.status);
    const data=await resp.json();
    drawOverlayBoxes([]);
    const ok=!!data.valid_date_iso;
    statusEl.textContent= ok?('Fecha detectada: '+data.valid_date_iso):'Fecha no detectada';
    statusEl.className='badge '+(ok?'ok':'err');
    const html = `
      <div class="card date">
        <b>Fecha actual:</b> ${nowISOish()}<br/>
        <b>Tipo:</b> Fecha de vencimiento<br/>
        <b>OCR:</b> ${data.raw_text||'—'}<br/>
        <b>Contexto:</b> ${data.context||'—'}<br/>
        <b>Fecha detectada:</b> ${data.matched_date||'—'}<br/>
        <b>Fecha válida (ISO):</b> ${data.valid_date_iso||'—'}<br/>
        <b>Guardado:</b> ${data.saved?'Sí':'No'} ${data.save_message?('— '+data.save_message):''}
      </div>`;
    expiryEl.innerHTML=html;
    if(data.saved) notify('success','Fecha guardada',data.valid_date_iso);
    else if(ok) notify('info','Fecha detectada','Confirma que corresponda al producto escaneado.');
    else notify('warning','No se detectó fecha','Intenta acercar la etiqueta al ROI.');
  }catch(e){
    statusEl.textContent='Error';
    statusEl.className='badge err';
    modal('error','Error detectando fecha',e.message);
  }
}

function daysUntil(isoDate){
  if(!isoDate) return null;
  const today=new Date(); today.setHours(0,0,0,0);
  const d=new Date(isoDate+"T00:00:00");
  return Math.ceil((d-today)/(1000*60*60*24));
}
function stateTag(days){
  if(days===null) return '';
  if(days<=5) return '<span class="tag red">≤ 5 días</span>';
  if(days<=10) return '<span class="tag yellow">≤ 10 días</span>';
  if(days>15) return '<span class="tag green">> 15 días</span>';
  return '';
}
function fmtTS(ts){ return ts ? ts.replace('T',' ').split('.')[0] : '—'; }

async function fetchProductos(){
  try{
    const resp=await fetch('/productos');
    if(!resp.ok) throw new Error('HTTP '+resp.status);
    const rows=await resp.json();
    tblBody.innerHTML=rows.map(r=>{
      const vence=r.fecha_vencimiento||'';
      const days=vence?daysUntil(vence):null;
      return `
        <tr>
          <td>${r.codigo}</td>
          <td class="muted">${r.tipo}</td>
          <td class="muted">${fmtTS(r.timestamp)}</td>
          <td>${vence||'—'}</td>
          <td>${stateTag(days)}</td>
        </tr>`;
    }).join('');
  }catch(e){
    console.error('fetch /productos', e);
    notify('error','Error cargando tabla',e.message);
  }
}

/* Cámara */
btnStart.onclick = async ()=>{
  try{
    stream=await navigator.mediaDevices.getUserMedia({ video:{facingMode:'environment'}, audio:false });
    video.srcObject=stream; await video.play();
    syncCanvasSizes(); drawOverlayBoxes([]); fixLayerHeight();
    window.addEventListener('resize',fixLayerHeight);
    setBarcodeMode();
    if(!poller){ await fetchProductos(); poller=setInterval(fetchProductos,3000);}
    btnStart.disabled=true; btnStop.disabled=false;
    notify('success','Cámara lista');
  }catch(e){ modal('error','No se pudo acceder a la cámara',e.message); }
};
btnStop.onclick = ()=>{
  if(timer){clearInterval(timer); timer=null;}
  if(poller){clearInterval(poller); poller=null;}
  if(progTimer){clearInterval(progTimer); progTimer=null;}
  if(stream){stream.getTracks().forEach(t=>t.stop()); stream=null;}
  btnStart.disabled=false; btnStop.disabled=true;
  statusEl.textContent='Idle'; statusEl.className='badge';
  ctxO.clearRect(0,0,overlay.width,overlay.height); layer.style.height='';
  liveStats.style.display='none';
  notify('info','Captura detenida');
};
btnClear.onclick = async ()=>{
  const c=await Swal.fire({ icon:'warning', title:'¿Limpiar tabla?', text:'Se eliminarán todos los productos registrados.', showCancelButton:true, confirmButtonText:'Sí, limpiar', cancelButtonText:'Cancelar' });
  if(!c.isConfirmed) return;
  try{
    const resp=await fetch('/clear',{method:'POST'});
    const data=await resp.json();
    if(!resp.ok||!data.ok) throw new Error(data.message||'No se pudo limpiar');
    resultsEl.innerHTML=''; expiryEl.innerHTML=''; lastEl.textContent='Último: —';
    await fetchProductos(); notify('success','Tabla limpiada',data.message||'');
  }catch(e){ modal('error','Error limpiando',e.message); }
};
btnMode.onclick = ()=>{ if (mode==='barcode') setExpiryMode(); else setBarcodeMode(); };
document.addEventListener('keydown', async (e)=>{
  if(e.key && e.key.toLowerCase()==='c'){
    if (mode!=='expiry') setExpiryMode();
    await captureExpiryOnce(); await fetchProductos();
    setTimeout(setBarcodeMode, 600);
  }
});

/* ====== Video Open/Closed (asíncrono con progreso) ====== */
vidFile.addEventListener('change', ()=>{ btnDetectVideo.disabled = !(vidFile.files && vidFile.files.length); });

// consola: barra ASCII
function asciiBar(pct, width=20){
  const filled = Math.round((pct/100) * width);
  return `[${'#'.repeat(filled)}${'.'.repeat(Math.max(0,width-filled))}] ${pct.toFixed(2)}%`;
}

function renderFinalVideoAndSegments(data){
  const { frames_total, frames_procesados, fps_est, fps_out, stride, segments, annotated_url_mp4, annotated_url_webm } = data;
  const ratio = frames_total ? Math.round((frames_procesados/frames_total)*100) : 0;

  detSummary.style.display = 'block';
  detSummary.innerHTML = `
    <b>Resumen del análisis</b><br/>
    <b>FPS fuente:</b> ${fps_est ?? '—'} &middot; <b>FPS salida:</b> ${fps_out ?? '—'} &middot; <b>Stride:</b> ${stride ?? '—'}<br/>
    <b>Frames totales:</b> ${frames_total} &middot; <b>Procesados:</b> ${frames_procesados} (${ratio}%)<br/>
    <b>Segmentos detectados:</b> ${segments ? segments.length : 0}
  `;

  detPreview.innerHTML = '';
  let anySource = false;
  if (annotated_url_mp4) {
    const s = document.createElement('source'); s.src = annotated_url_mp4; s.type = 'video/mp4';
    detPreview.appendChild(s); anySource = true;
    detDownloadMp4.href = annotated_url_mp4; detDownloadMp4.style.display='';
  } else { detDownloadMp4.removeAttribute('href'); detDownloadMp4.style.display='none'; }
  if (annotated_url_webm) {
    const s = document.createElement('source'); s.src = annotated_url_webm; s.type = 'video/webm';
    detPreview.appendChild(s); anySource = true;
    detDownloadWebm.href = annotated_url_webm; detDownloadWebm.style.display='';
  } else { detDownloadWebm.removeAttribute('href'); detDownloadWebm.style.display='none'; }
  if (anySource) { detMedia.style.display='block'; detPreview.load(); } else { detMedia.style.display='none'; }

  if (segments && segments.length){
    detSegments.style.display = 'block';
    segTable.innerHTML = `
      <table style="width:100%; border-collapse:collapse; font-size:13px;">
        <thead>
          <tr>
            <th style="text-align:left;">#</th>
            <th style="text-align:left;">Estado</th>
            <th style="text-align:left;">Inicio (s)</th>
            <th style="text-align:left;">Fin (s)</th>
            <th style="text-align:left;">Duración (s)</th>
            <th style="text-align:left;">Conf.</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          ${segments.map((s,idx)=>`
            <tr>
              <td>${idx+1}</td>
              <td>${s.label === 'abierto'
                    ? '<span class="tag yellow">ABIERTO</span>'
                    : '<span class="tag green">CERRADO</span>'}</td>
              <td>${s.start_time}</td>
              <td>${s.end_time}</td>
              <td>${s.duration_s}</td>
              <td>${s.avg_prob}</td>
              <td><button class="btn-go" data-t="${s.start_time}">Ver</button></td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    `;
    segTable.querySelectorAll('.btn-go').forEach(btn=>{
      btn.addEventListener('click', ()=>{
        const t = parseFloat(btn.getAttribute('data-t')||'0');
        if (!isNaN(t) && detPreview) {
          detPreview.currentTime = t;
          detPreview.play().catch(()=>{});
          detMedia.scrollIntoView({behavior:'smooth', block:'nearest'});
        }
      });
    });
  } else {
    detSegments.style.display = 'none';
    segTable.innerHTML = '';
  }
}

function startProgress(jobId){
  liveStats.style.display = 'block';
  openPctEl.textContent = '0%'; closedPctEl.textContent = '0%';
  doneEl.textContent = '0'; totalEl.textContent = '0';

  if (progTimer) { clearInterval(progTimer); progTimer = null; }
  progTimer = setInterval(async ()=>{
    try{
      const r = await fetch(`/openclosed/video_progress?job_id=${jobId}`);
      const p = await r.json();
      if (!r.ok) throw new Error(p.error || 'progress error');
      if (p.error){ throw new Error(p.error); }

      doneEl.textContent  = p.done;
      totalEl.textContent = p.total || 0;
      openPctEl.textContent   = `${p.pct_open}%`;
      closedPctEl.textContent = `${p.pct_closed}%`;

      // Log en consola del navegador
      const pctDone = p.total ? (p.done / p.total) * 100 : 0;
      console.log(
        `Frames ${p.done}/${p.total} ${asciiBar(pctDone)} | ` +
        `ABIERTO: ${p.pct_open}% · CERRADO: ${p.pct_closed}%`
      );

      if (p.ready){
        clearInterval(progTimer); progTimer = null;
        const rr = await fetch(`/openclosed/video_result?job_id=${jobId}`);
        const data = await rr.json();
        if (!rr.ok) throw new Error(data.error || 'result error');
        renderFinalVideoAndSegments(data);
        liveStats.style.display = 'none';
        Swal.close();
        notify('success','Análisis terminado');
      }
    }catch(e){
      clearInterval(progTimer); progTimer=null;
      liveStats.style.display = 'none';
      Swal.close();
      modal('error','Error en el análisis', e.message);
    }
  }, 300);
}

btnDetectVideo.onclick = async ()=>{
  const file = vidFile.files && vidFile.files[0];
  if(!file) return;
  const stride = Math.max(1, parseInt(strideEl.value||'10',10));

  Swal.fire({ title:'Analizando video…', html:'El modelo está trabajando.', allowOutsideClick:false, didOpen:()=>{ Swal.showLoading(); } });

  try {
    const form = new FormData();
    form.append('video', file);
    form.append('stride', String(stride));
    const init = await fetch('/openclosed/video_async', { method:'POST', body: form });
    const respInit = await init.json();
    if (!init.ok || !respInit.job_id) throw new Error(respInit.error || 'no job id');
    startProgress(respInit.job_id);
  } catch(e) {
    Swal.close();
    modal('error','No se pudo iniciar el análisis', e.message);
  }
};
</script>
</body>
</html>
"""

# =========================
# Rutas
# =========================
@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=INDEX_HTML, status_code=200)

@app.get("/productos")
async def get_productos():
    try:
        with engine.begin() as conn:
            stmt = (
                select(
                    productos.c.codigo,
                    productos.c.tipo,
                    productos.c.timestamp,
                    productos.c.fecha_vencimiento
                )
                .order_by(nulls_last(productos.c.fecha_vencimiento.asc()))
            )
            rows = conn.execute(stmt).all()

        def row_to_json(r):
            ts = r.timestamp.isoformat() if r.timestamp else None
            fv = r.fecha_vencimiento.isoformat() if r.fecha_vencimiento else None
            return {"codigo": int(r.codigo) if r.codigo is not None else None,
                    "tipo": r.tipo, "timestamp": ts, "fecha_vencimiento": fv}

        return JSONResponse([row_to_json(r) for r in rows])
    except SQLAlchemyError as e:
        return JSONResponse({"error": f"DB: {str(e)}"}, status_code=500)

@app.post("/decode")
async def decode(request: Request):
    form = await request.form()
    file = form.get("image")
    if file is None:
        return JSONResponse({"results": [], "error": "no-image"}, status_code=400)
    content = await file.read()
    results = decode_image_bytes_with_roi(content)
    return JSONResponse({"results": results})

@app.post("/expiry")
async def expiry(request: Request):
    form = await request.form()
    file = form.get("image")
    if file is None:
        return JSONResponse({"error": "no-image"}, status_code=400)
    content = await file.read()
    result = detect_expiry_in_image(content)

    saved = False
    save_message = ""
    if result.get("valid_date_iso"):
        ok, msg = guardar_fecha_vencimiento_en_ultima_fila(result["valid_date_iso"])
        saved, save_message = ok, msg
    else:
        save_message = "No se detectó fecha válida."
    result.update({"saved": saved, "save_message": save_message})
    return JSONResponse(result)

@app.post("/clear")
async def clear_table():
    try:
        codigos_detectados_sesion.clear()
        ok, msg = limpiar_db()
        status = 200 if ok else 500
        return JSONResponse({"ok": ok, "message": msg}, status_code=status)
    except Exception as e:
        return JSONResponse({"ok": False, "message": f"Error: {e}"}, status_code=500)

# ===== Endpoint síncrono (compatibilidad) =====
@app.post("/openclosed/video")
async def open_closed_from_video(request: Request):
    form = await request.form()
    file = form.get("video")
    stride = int(form.get("stride") or 10)
    if file is None:
        return JSONResponse({"error": "no-video"}, status_code=400)

    has_ffmpeg = shutil.which("ffmpeg") is not None
    ts_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    tmp_in = os.path.join("static", f"_upload_{ts_tag}.mp4")
    tmp_avi = os.path.join("static", f"_annot_{ts_tag}.avi")
    out_mp4 = os.path.join("static", f"annotated_{ts_tag}.mp4")
    out_webm = os.path.join("static", f"annotated_{ts_tag}.webm")

    try:
        content = await file.read()
        with open(tmp_in, "wb") as f: f.write(content)

        cap = cv.VideoCapture(tmp_in)
        if not cap.isOpened():
            return JSONResponse({"error": "No se pudo abrir el video"}, status_code=400)

        frames_total = int(cap.get(cv.CAP_PROP_FRAME_COUNT)) or 0
        fps_est = float(cap.get(cv.CAP_PROP_FPS)) if cap.get(cv.CAP_PROP_FPS) else 25.0
        fps_out = max(1.0, fps_est / max(1, stride))

        i = 0; frames_proc = 0; timeline = []; writer = None
        fourcc = cv.VideoWriter_fourcc(*'MJPG')

        while True:
            ret, frame = cap.read()
            if not ret: break
            if stride > 1 and (i % stride != 0):
                i += 1; continue

            label, conf, cropped = predict_open_closed(frame)
            if writer is None:
                h, w = cropped.shape[:2]
                writer = cv.VideoWriter(tmp_avi, fourcc, fps_out, (w, h))

            text = f"{label.upper()} ({conf:.2f})"
            color = (0, 255, 0) if label == "cerrado" else (0, 0, 255)
            cv.putText(cropped, text, (30, 40), cv.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv.LINE_AA)
            cv.rectangle(cropped, (20, 15), (30, 25), color, -1)

            writer.write(cropped)
            timeline.append({"i_frame": i, "label": label, "prob": float(conf)})
            frames_proc += 1; i += 1

        cap.release()
        if writer is not None: writer.release()

        # Segmentos contiguos
        segments = []
        if timeline:
            run_label = timeline[0]["label"]; run_start = timeline[0]["i_frame"]; probs=[timeline[0]["prob"]]
            for t in timeline[1:]:
                if t["label"] == run_label:
                    probs.append(t["prob"])
                else:
                    run_end = t["i_frame"] - 1
                    t0 = run_start / fps_est; t1 = run_end / fps_est
                    segments.append({
                        "label": run_label, "start_frame": run_start, "end_frame": run_end,
                        "start_time": round(t0,2), "end_time": round(t1,2),
                        "duration_s": round(max(0, t1 - t0),2), "avg_prob": round(sum(probs)/max(1,len(probs)),3)
                    })
                    run_label = t["label"]; run_start = t["i_frame"]; probs=[t["prob"]]
            run_end = timeline[-1]["i_frame"]; t0 = run_start / fps_est; t1 = run_end / fps_est
            segments.append({
                "label": run_label, "start_frame": run_start, "end_frame": run_end,
                "start_time": round(t0,2), "end_time": round(t1,2),
                "duration_s": round(max(0, t1 - t0),2), "avg_prob": round(sum(probs)/max(1,len(probs)),3)
            })

        annotated_url_mp4 = None; annotated_url_webm = None
        if has_ffmpeg and os.path.exists(tmp_avi):
            try:
                subprocess.run(["ffmpeg","-y","-i",tmp_avi,"-c:v","libx264","-preset","veryfast","-crf","23","-movflags","+faststart",out_mp4],
                               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if os.path.exists(out_mp4): annotated_url_mp4 = f"/static/{os.path.basename(out_mp4)}"
            except Exception: pass
            try:
                subprocess.run(["ffmpeg","-y","-i",tmp_avi,"-c:v","libvpx-vp9","-b:v","1M",out_webm],
                               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if os.path.exists(out_webm): annotated_url_webm = f"/static/{os.path.basename(out_webm)}"
            except Exception: pass

        if not annotated_url_mp4 and os.path.exists(tmp_avi):
            annotated_url_mp4 = f"/static/{os.path.basename(tmp_avi)}"

        return JSONResponse({
            "frames_total": frames_total, "frames_procesados": frames_proc,
            "fps_est": fps_est, "fps_out": fps_out, "stride": stride,
            "timeline": timeline, "segments": segments,
            "annotated_url_mp4": annotated_url_mp4, "annotated_url_webm": annotated_url_webm,
            "ffmpeg": has_ffmpeg, "model_loaded": _model_loaded, "model_error": _model_error
        })
    except Exception as e:
        return JSONResponse({"error": f"Error procesando video: {e}"}, status_code=500)
    finally:
        try:
            if os.path.exists(tmp_in): os.remove(tmp_in)
        except Exception: pass

# ===== Detección asíncrona + progreso =====
@app.post("/openclosed/video_async")
async def openclosed_video_async(request: Request):
    form = await request.form()
    file = form.get("video")
    stride = int(form.get("stride") or 10)
    if file is None:
        return JSONResponse({"error": "no-video"}, status_code=400)

    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"total":0,"done":0,"open":0,"closed":0,"ready":False,"result":None,"error":None}

    in_path = os.path.join("static", f"_upload_{job_id}.mp4")
    content = await file.read()
    with open(in_path, "wb") as f:
        f.write(content)

    Thread(target=_process_video_job, args=(job_id, in_path, stride), daemon=True).start()
    return JSONResponse({"job_id": job_id})

@app.get("/openclosed/video_progress")
async def openclosed_video_progress(job_id: str):
    j = JOBS.get(job_id)
    if not j:
        return JSONResponse({"error":"job not found"}, status_code=404)
    total = j["total"] or 1
    done = j["done"]
    abierto = j["open"]
    cerrado = j["closed"]
    return JSONResponse({
        "ready": j["ready"],
        "error": j["error"],
        "done": done,
        "total": j["total"],
        "pct_open": round(100*abierto/max(1, done), 2) if done else 0.0,
        "pct_closed": round(100*cerrado/max(1, done), 2) if done else 0.0
    })

@app.get("/openclosed/video_result")
async def openclosed_video_result(job_id: str):
    j = JOBS.get(job_id)
    if not j:
        return JSONResponse({"error":"job not found"}, status_code=404)
    if j["error"]:
        return JSONResponse({"error": j["error"]}, status_code=500)
    if not j["ready"]:
        return JSONResponse({"ready": False})
    return JSONResponse(j["result"])

# =========================
# Nota de ejecución
# =========================
# uvicorn app:app --reload --host 0.0.0.0 --port 8000
