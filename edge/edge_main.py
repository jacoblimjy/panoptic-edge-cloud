import os, time, json, io, psutil
from typing import Optional, Tuple
import requests
import cv2
from ultralytics import YOLO
import numpy as np
import mediapipe as mp


CLOUD_HOST = os.getenv("CLOUD_HOST", "cloud")
CLOUD_PORT = int(os.getenv("CLOUD_PORT", "5000"))
CLOUD_URL  = f"http://{CLOUD_HOST}:{CLOUD_PORT}/analyze"

INPUT_SOURCE = os.getenv("INPUT_SOURCE", "file")  
VIDEO_PATH   = os.getenv("VIDEO_PATH", "/app/streaming/sample.mp4")
RTSP_URL     = os.getenv("RTSP_URL", "rtsp://mediamtx:8554/live.sdp")

EDGE_MAX_FPS = float(os.getenv("EDGE_MAX_FPS", "5"))
PERSON_CONF  = float(os.getenv("PERSON_CONF", "0.4"))
SOURCE_ID    = os.getenv("SOURCE_ID", "cam01")

EDGE_ACTIVITY = os.getenv("EDGE_ACTIVITY", "false").lower() == "true"  
OFFLOAD_TO_CLOUD = os.getenv("OFFLOAD_TO_CLOUD", "true").lower() == "true" 

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
EDGE_LOG = os.path.join(RESULTS_DIR, "metrics_edge.jsonl")


def jlog(**kv):
    rec = {"ts": time.time(), **kv}
    line = json.dumps(rec, ensure_ascii=False)
    print(f"[EDGE] {line}", flush=True)
    with open(EDGE_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def open_capture() -> cv2.VideoCapture:
    if INPUT_SOURCE == "webcam":
        return cv2.VideoCapture(0)
    if INPUT_SOURCE == "rtsp":
        return cv2.VideoCapture(RTSP_URL)
    return cv2.VideoCapture(VIDEO_PATH)

def jpeg_bytes(img_bgr) -> bytes:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return buf.tobytes() if ok else b""


MODEL_NAME = os.getenv("YOLO_MODEL", "yolov8n.pt")
model = YOLO(MODEL_NAME)

mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose(static_image_mode=True, model_complexity=1)

def run_pose_on_crop(img_bgr: np.ndarray, bbox: Tuple[int,int,int,int]) -> Tuple[Optional[str], Optional[float], int]:
    # bbox = (x1,y1,x2,y2), crop & pad a bit
    x1,y1,x2,y2 = bbox
    h, w = img_bgr.shape[:2]
    dx = int(0.1*(x2-x1)); dy = int(0.1*(y2-y1))
    x1 = max(0, x1-dx); y1 = max(0, y1-dy)
    x2 = min(w-1, x2+dx); y2 = min(h-1, y2+dy)
    crop = img_bgr[y1:y2, x1:x2]
    t0 = time.time()
    res = pose_model.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    t1 = time.time()
    if not res.pose_landmarks:
        return "unknown", None, int((t1-t0)*1000)
    lm = res.pose_landmarks.landmark
    # torso angle: mid-shoulder -> mid-hip vs vertical
    try:
        ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]; rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        lh = lm[mp_pose.PoseLandmark.LEFT_HIP]; rh = lm[mp_pose.PoseLandmark.RIGHT_HIP]
        if min(ls.visibility, rs.visibility, lh.visibility, rh.visibility) < 0.4:
            return "person_detected", None, int((t1-t0)*1000)
        sx, sy = (ls.x+rs.x)/2.0, (ls.y+rs.y)/2.0
        hx, hy = (lh.x+rh.x)/2.0, (lh.y+rh.y)/2.0
        # use crop dims for stability
        dxp = (sx - hx) * crop.shape[1]
        dyp = (sy - hy) * crop.shape[0]
        angle_deg = float(np.degrees(np.arctan2(abs(dxp), abs(dyp)+1e-6)))
        if angle_deg < 30: activity = "standing_or_walking"
        elif angle_deg > 55: activity = "falling_or_lying"
        else: activity = "uncertain_posture"
        return activity, angle_deg, int((t1-t0)*1000)
    except Exception:
        return "person_detected", None, int((t1-t0)*1000)

def send_to_cloud(img_bgr, captured_ts: float) -> Optional[dict]:
    data = jpeg_bytes(img_bgr)
    if not data:
        return None
    headers = {"X-Source-Id": SOURCE_ID, "X-Captured-Ts": str(captured_ts)}
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
            return {"cloud_ms": int((t_resp - t_send)*1000), "resp": resp.json()}
        jlog(stage="edge", event="cloud_http_error", status=resp.status_code)
        return None
    except Exception as e:
        jlog(stage="edge", event="cloud_exception", err=str(e))
        return None

# -----------------------
# Main
# -----------------------
def main():
    jlog(stage="edge", event="boot",
         input_source=INPUT_SOURCE, video_path=VIDEO_PATH, rtsp=RTSP_URL,
         cloud=CLOUD_URL, max_fps=EDGE_MAX_FPS,
         edge_activity=EDGE_ACTIVITY, offload_to_cloud=OFFLOAD_TO_CLOUD)

    # health ping
    try:
        h = requests.get(f"http://{CLOUD_HOST}:{CLOUD_PORT}/health", timeout=2)
        jlog(stage="edge", event="cloud_health", ok=h.ok)
    except Exception as e:
        jlog(stage="edge", event="cloud_health_fail", err=str(e))

    cap = open_capture()
    if not cap.isOpened():
        jlog(stage="edge", event="open_capture_fail", source=INPUT_SOURCE, path=VIDEO_PATH)
        time.sleep(3); return

    min_dt = 1.0 / max(EDGE_MAX_FPS, 0.1)
    last_proc = 0.0
    ema_fps = None

    while True:
        t_cap0 = time.time()
        ok, frame = cap.read()
        if not ok or frame is None:
            if INPUT_SOURCE == "file":
                cap.release(); cap = open_capture(); ok, frame = cap.read()
                if not ok: jlog(stage="edge", event="end_of_file"); break
            else:
                jlog(stage="edge", event="capture_error"); time.sleep(0.1); continue

        h, w = frame.shape[:2]
        if w > 960:
            frame = cv2.resize(frame, (960, int(960*h/w)))

        now = time.time()
        if (now - last_proc) < min_dt:
            continue
        last_proc = now

        sys_cpu = psutil.cpu_percent(interval=None); sys_mem = psutil.virtual_memory().percent

        t_inf0 = time.time()
        results = model(frame, conf=PERSON_CONF, classes=[0], verbose=False)
        t_inf1 = time.time()

        persons = []
        for r in results:
            if r.boxes is None: continue
            for b in r.boxes:
                cls = int(b.cls.item()); conf = float(b.conf.item())
                if cls == 0 and conf >= PERSON_CONF:
                    x1,y1,x2,y2 = map(lambda x:int(x), b.xyxy[0].tolist())
                    persons.append((x1,y1,x2,y2, conf))

        activity_edge, angle_edge, pose_ms = None, None, None
        if EDGE_ACTIVITY and persons:
            bb = max(persons, key=lambda z:(z[2]-z[0])*(z[3]-z[1]))
            activity_edge, angle_edge, pose_ms = run_pose_on_crop(frame, bb[:4])

        activity_cloud, alert_cloud, cloud_ms = None, None, None
        if OFFLOAD_TO_CLOUD and persons:
            vis = frame.copy()
            for (x1,y1,x2,y2,conf) in persons:
                cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(vis, f"person {conf:.2f}", (x1, max(20, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            sent = send_to_cloud(vis, t_cap0)
            if sent and sent.get("resp"):
                activity_cloud = sent["resp"].get("activity")
                alert_cloud = sent["resp"].get("alert")
                cloud_ms = sent.get("cloud_ms")

        inst_fps = 1.0 / max((time.time() - t_inf0), 1e-6)
        ema_fps = inst_fps if ema_fps is None else (0.9*ema_fps + 0.1*inst_fps)

        rec = {
            "stage": "edge", "mode": ("edge_only" if EDGE_ACTIVITY and not OFFLOAD_TO_CLOUD else "edge_cloud"),
            "source": INPUT_SOURCE, "ts": t_cap0,
            "infer_ms": int((t_inf1 - t_inf0)*1000),
            "pose_ms": pose_ms, "torso_deg_edge": angle_edge,
            "fps_inst": round(inst_fps, 2), "fps_ema": round(ema_fps, 2),
            "cpu": sys_cpu, "mem": sys_mem,
            "persons": len(persons),
            "activity_edge": activity_edge,
            "activity_cloud": activity_cloud, "alert_cloud": alert_cloud,
            "cloud_ms": cloud_ms,
            # handy approximations:
            "e2e_ms_local": (int((t_inf1 - t_inf0)*1000) + (pose_ms or 0)) if EDGE_ACTIVITY else None,
            "e2e_ms_cloud": (int((t_inf1 - t_inf0)*1000) + (cloud_ms or 0)) if OFFLOAD_TO_CLOUD else None
        }
        jlog(**rec)

    cap.release()
    jlog(stage="edge", event="shutdown")

if __name__ == "__main__":
    main()
