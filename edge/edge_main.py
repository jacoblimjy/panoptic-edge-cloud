import os, time, json, io, base64, psutil, threading
from typing import Optional
import requests
import cv2
from ultralytics import YOLO

# -----------------------
# ENV / Config
# -----------------------
CLOUD_HOST = os.getenv("CLOUD_HOST", "cloud")
CLOUD_PORT = int(os.getenv("CLOUD_PORT", "5000"))
CLOUD_URL  = f"http://{CLOUD_HOST}:{CLOUD_PORT}/analyze"

INPUT_SOURCE = os.getenv("INPUT_SOURCE", "file")  # webcam | file | rtsp
VIDEO_PATH   = os.getenv("VIDEO_PATH", "/app/streaming/sample.mp4")
RTSP_URL     = os.getenv("RTSP_URL", "rtsp://host.docker.internal:8554/live.sdp")

EDGE_MAX_FPS = float(os.getenv("EDGE_MAX_FPS", "5"))   # cap processing rate
PERSON_CONF  = float(os.getenv("PERSON_CONF", "0.4"))  # YOLO confidence
SEND_WHEN_NO_PERSON = os.getenv("SEND_WHEN_NO_PERSON", "false").lower() == "true"
SOURCE_ID    = os.getenv("SOURCE_ID", "cam01")

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
EDGE_LOG = os.path.join(RESULTS_DIR, "metrics_edge.jsonl")

# -----------------------
# Utils
# -----------------------
def jlog(**kv):
    rec = {"ts": time.time(), **kv}
    line = json.dumps(rec, ensure_ascii=False)
    print(f"[EDGE] {line}", flush=True)
    with open(EDGE_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def open_capture() -> cv2.VideoCapture:
    if INPUT_SOURCE == "webcam":
        cap = cv2.VideoCapture(0)
    elif INPUT_SOURCE == "rtsp":
        cap = cv2.VideoCapture(RTSP_URL)
    else:
        cap = cv2.VideoCapture(VIDEO_PATH)
    return cap

def jpeg_bytes(img_bgr) -> bytes:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return buf.tobytes() if ok else b""

# -----------------------
# Detector setup
# -----------------------
# class 0 in COCO = person
MODEL_NAME = os.getenv("YOLO_MODEL", "yolov8n.pt")  # auto-downloaded
model = YOLO(MODEL_NAME)  # loads weights

# -----------------------
# Offload to cloud
# -----------------------
def send_to_cloud(img_bgr, captured_ts: float) -> Optional[dict]:
    data = jpeg_bytes(img_bgr)
    if not data:
        return None
    headers = {
        "X-Source-Id": SOURCE_ID,
        "X-Captured-Ts": str(captured_ts)
    }
    try:
        t_send = time.time()
        resp = requests.post(
            CLOUD_URL,
            files={"image": ("frame.jpg", io.BytesIO(data), "image/jpeg")},
            headers=headers,
            timeout=5
        )
        t_resp = time.time()
        if resp.ok:
            out = resp.json()
            return {"cloud_ms": int((t_resp - t_send)*1000), "resp": out}
        else:
            jlog(stage="edge", event="cloud_http_error", status=resp.status_code)
            return None
    except Exception as e:
        jlog(stage="edge", event="cloud_exception", err=str(e))
        return None

# -----------------------
# Main loop
# -----------------------
def main():
    jlog(stage="edge", event="boot",
         input_source=INPUT_SOURCE, video_path=VIDEO_PATH, rtsp=RTSP_URL,
         cloud=CLOUD_URL, max_fps=EDGE_MAX_FPS)

    # quick cloud health ping
    try:
        h = requests.get(f"http://{CLOUD_HOST}:{CLOUD_PORT}/health", timeout=3)
        jlog(stage="edge", event="cloud_health", ok=h.ok)
    except Exception as e:
        jlog(stage="edge", event="cloud_health_fail", err=str(e))

    cap = open_capture()
    if not cap.isOpened():
        jlog(stage="edge", event="open_capture_fail", source=INPUT_SOURCE, path=VIDEO_PATH)
        time.sleep(3)
        return

    min_dt = 1.0 / max(EDGE_MAX_FPS, 0.1)
    last_proc = 0.0
    ema_fps = None

    while True:
        t_cap0 = time.time()
        ok, frame = cap.read()
        if not ok or frame is None:
            # Loop video files
            if INPUT_SOURCE == "file":
                cap.release()
                cap = open_capture()
                ok, frame = cap.read()
                if not ok:
                    jlog(stage="edge", event="end_of_file")
                    break
            else:
                jlog(stage="edge", event="capture_error")
                time.sleep(0.1)
                continue

        # Resize for speed (keep aspect)
        h, w = frame.shape[:2]
        if w > 960:
            frame = cv2.resize(frame, (960, int(960*h/w)))

        # FPS cap (smart sampling)
        now = time.time()
        if (now - last_proc) < min_dt:
            # skip heavy work, but keep loop responsive
            continue
        last_proc = now

        # Metrics before inference
        sys_cpu = psutil.cpu_percent(interval=None)
        sys_mem = psutil.virtual_memory().percent

        # Inference
        t_inf0 = time.time()
        results = model(frame, conf=PERSON_CONF, classes=[0], verbose=False)  # only 'person'
        t_inf1 = time.time()

        # Parse detections
        persons = []
        for r in results:
            if r.boxes is None: 
                continue
            for b in r.boxes:
                cls = int(b.cls.item())
                conf = float(b.conf.item())
                if cls == 0 and conf >= PERSON_CONF:
                    x1,y1,x2,y2 = map(lambda x:int(x), b.xyxy[0].tolist())
                    persons.append((x1,y1,x2,y2, conf))

        # If persons detected (or SEND_WHEN_NO_PERSON), offload one annotated frame
        sent = None
        activity = None
        alert = None
        t_cloud_ms = None

        if persons or SEND_WHEN_NO_PERSON:
            # draw boxes for context
            vis = frame.copy()
            for (x1,y1,x2,y2,conf) in persons:
                cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(vis, f"person {conf:.2f}", (x1, max(20, y1-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            sent = send_to_cloud(vis, t_cap0)
            if sent and sent.get("resp"):
                activity = sent["resp"].get("activity")
                alert = sent["resp"].get("alert")
                t_cloud_ms = sent.get("cloud_ms")

        # Update FPS EMA
        inst_fps = 1.0 / max((time.time() - t_inf0), 1e-6)
        ema_fps = inst_fps if ema_fps is None else (0.9*ema_fps + 0.1*inst_fps)

        # Log JSONL
        rec = {
            "stage": "edge",
            "source": INPUT_SOURCE,
            "ts": t_cap0,
            "infer_ms": int((t_inf1 - t_inf0)*1000),
            "fps_inst": round(inst_fps, 2),
            "fps_ema": round(ema_fps, 2),
            "cpu": sys_cpu,
            "mem": sys_mem,
            "persons": len(persons),
            "cloud_ms": t_cloud_ms,
            "activity": activity,
            "alert": alert
        }
        jlog(**rec)

        # optional keyboard exit if we ever attach a display (not in container)
        # if cv2.waitKey(1) == 27: break

    cap.release()
    jlog(stage="edge", event="shutdown")

if __name__ == "__main__":
    main()
