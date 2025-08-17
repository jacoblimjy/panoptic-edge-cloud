import os, io, time, json, base64
from datetime import datetime, timedelta, timezone
from collections import deque, defaultdict

import numpy as np
import cv2
from flask import Flask, jsonify, request
import mediapipe as mp

RESULTS_DIR = os.environ.get("RESULTS_DIR", "/app/results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SAVE_ANNOTATIONS = os.environ.get("SAVE_ANNOTATIONS", "false").lower() == "true"
LOITER_WINDOW_S = int(os.environ.get("LOITER_WINDOW_S", "60"))
LOITER_RADIUS = float(os.environ.get("LOITER_RADIUS", "0.03"))
FALL_PERSIST_S = int(os.environ.get("FALL_PERSIST_S", "10"))
NIGHT_START_HOUR = int(os.environ.get("NIGHT_START_HOUR", "22"))
NIGHT_END_HOUR = int(os.environ.get("NIGHT_END_HOUR", "6"))
SGT = timezone(timedelta(hours=8))

app = Flask(__name__)
mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose(static_image_mode=True, model_complexity=1)

state = defaultdict(lambda: {"positions": deque(maxlen=1200), "fall_start": None, "last_alert_ts": 0.0})

def is_night(ts_s: float) -> bool:
    dt = datetime.fromtimestamp(ts_s, SGT)
    h = dt.hour
    if NIGHT_START_HOUR > NIGHT_END_HOUR:
        return h >= NIGHT_START_HOUR or h < NIGHT_END_HOUR
    else:
        return NIGHT_START_HOUR <= h < NIGHT_END_HOUR

def write_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def decode_image_from_request(req):
    if "image" in req.files:
        data = req.files["image"].read()
    elif req.mimetype == "application/octet-stream":
        data = req.get_data()
    else:
        js = req.get_json(silent=True) or {}
        b64 = js.get("image_b64")
        if not b64:
            return None
        data = base64.b64decode(b64)
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def torso_angle_degrees(lm, image_w, image_h):
    try:
        ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        lh = lm[mp_pose.PoseLandmark.LEFT_HIP]
        rh = lm[mp_pose.PoseLandmark.RIGHT_HIP]
    except Exception:
        return None
    if min(ls.visibility, rs.visibility, lh.visibility, rh.visibility) < 0.4:
        return None
    sx, sy = (ls.x + rs.x) * 0.5, (ls.y + rs.y) * 0.5
    hx, hy = (lh.x + rh.x) * 0.5, (lh.y + rh.y) * 0.5
    dx = (sx - hx) * image_w
    dy = (sy - hy) * image_h
    return float(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6)))

def person_centroid(lm):
    vis_pts = [(p.x, p.y) for p in lm if p.visibility >= 0.4]
    if not vis_pts:
        return None
    xs, ys = zip(*vis_pts)
    return (float(np.mean(xs)), float(np.mean(ys)))

def annotate_image(img_bgr, lm, activity, alert=None):
    h, w = img_bgr.shape[:2]
    if lm:
        for p in lm:
            if p.visibility < 0.4: continue
            cx, cy = int(p.x * w), int(p.y * h)
            cv2.circle(img_bgr, (cx, cy), 3, (0, 255, 0), -1)
    title = f"{activity}" + (f" | {alert}" if alert else "")
    cv2.putText(img_bgr, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255) if alert else (255,255,255), 2, cv2.LINE_AA)
    return img_bgr

@app.get("/health")
def health():
    return jsonify({"ok": True, "service": "cloud", "msg": "cloud up"}), 200

@app.post("/analyze")
def analyze():
    t0 = time.time()
    src = request.headers.get("X-Source-Id", "default")
    img_bgr = decode_image_from_request(request)
    if img_bgr is None:
        return jsonify({"error": "no image provided"}), 400

    h, w = img_bgr.shape[:2]
    t1 = time.time()
    result = pose_model.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    t2 = time.time()

    activity, alert = "unknown", None
    torso_deg, centroid = None, None
    night = is_night(t0)

    if result.pose_landmarks:
        lm = result.pose_landmarks.landmark
        torso_deg = torso_angle_degrees(lm, w, h)
        centroid = person_centroid(lm)
        if torso_deg is not None:
            if torso_deg < 30: activity = "standing_or_walking"
            elif torso_deg > 55: activity = "falling_or_lying"
            else: activity = "uncertain_posture"
        else:
            activity = "person_detected"

        st = state[src]
        if centroid is not None:
            st["positions"].append((t0, centroid))
            window = [(ts, xy) for ts, xy in st["positions"] if t0 - ts <= LOITER_WINDOW_S]
            if len(window) >= 5:
                xs = [xy[0] for _, xy in window]; ys = [xy[1] for _, xy in window]
                meanx, meany = np.mean(xs), np.mean(ys)
                rms = float(np.sqrt(np.mean([(x-meanx)**2 + (y-meany)**2 for x,y in zip(xs, ys)])))
            else:
                rms = None
        else:
            rms = None

        if activity == "falling_or_lying":
            if st["fall_start"] is None: st["fall_start"] = t0
            fall_elapsed = t0 - st["fall_start"]
        else:
            fall_elapsed = 0.0
            st["fall_start"] = None

        may_alert = (t0 - st["last_alert_ts"]) > 30.0
        if activity == "falling_or_lying" and fall_elapsed >= FALL_PERSIST_S and may_alert:
            alert = "ALERT_FALL_PERSIST"; st["last_alert_ts"] = t0
        if night and rms is not None and rms < LOITER_RADIUS and may_alert:
            alert = "ALERT_NIGHT_LOITERING"; st["last_alert_ts"] = t0

        if SAVE_ANNOTATIONS:
            annotated = annotate_image(img_bgr.copy(), lm, activity, alert)
            ts_str = datetime.fromtimestamp(t0, SGT).strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(os.path.join(RESULTS_DIR, f"annot_{ts_str}_{src}.jpg"), annotated)

    recv_ms = int((t1 - t0)*1000)
    infer_ms = int((t2 - t1)*1000)
    total_ms = int((time.time() - t0)*1000)
    resp = {
        "activity": activity,
        "alert": alert,
        "metrics": {
            "recv_ms": recv_ms, "infer_ms": infer_ms, "total_ms": total_ms,
            "torso_deg": torso_deg, "centroid": centroid, "night": night
        }
    }
    write_jsonl(os.path.join(RESULTS_DIR, "metrics_cloud.jsonl"), {"ts": t0, "src": src, "stage": "cloud", **resp})
    return jsonify(resp), 200

if __name__ == "__main__":
    # Important: no debug/reloader so the process stays in foreground
    app.run(host="0.0.0.0", port=5000)
