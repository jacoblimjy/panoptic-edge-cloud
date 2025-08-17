
# Real-Time Edge–Cloud Person Activity Recognition (Panoptic)

Division / Team: Content & Insights Management Program / Panoptic

## Overview

A two-tier, production-style system that processes live or recorded video in near real time:

* **Edge** container: runs lightweight person detection (YOLOv8n), smart frame sampling, optional local activity classification, metrics logging, and selective offload to cloud.
* **Cloud** container: runs visual activity analysis (MediaPipe Pose) and generates contextual alerts based on time patterns and activity combinations.
* **Monitoring** tools: parse JSONL logs into CSVs/plots and compute latency/FPS/CPU summaries (including end-to-end).
* **Streaming** tools: optional RTSP server + feeder to simulate IP camera streams.

Targets **< 15 s end-to-end** latency; typical runs on a laptop are **\~0.2 s p95** per analyzed frame.

---

## Features

* Real-time person detection (YOLOv8n CPU).
* Smart sampling (FPS cap) to manage edge resources.
* Edge→Cloud selective offload of annotated frames.
* Cloud activity classification via MediaPipe Pose (standing/walking vs falling/lying) with contextual alerts:

  * Persistent falls.
  * Night-time loitering (centroid movement).
* Detailed metrics:

  * Edge: FPS, inference ms, CPU/MEM, offload RTT.
  * Cloud: receive/infer/total ms, activity, alerts.
* Analyzer outputs CSVs, plots, p50/p95 latency, and **end-to-end** latency estimates.
* Works with **file**, **RTSP**, or **webcam** (Linux).

---

## Requirements

* Docker Desktop (macOS/Windows/Linux)
* Git (for commits)
* A small MP4 at `streaming/sample.mp4`

> macOS is fully supported using file or RTSP input. Webcam `/dev/video0` is Linux-only.

---

## Repository Structure

```
edge/          YOLOv8 person detection; optional local pose; offload; metrics
cloud/         Flask app; MediaPipe Pose; contextual alerts; metrics
monitoring/    Analyzer to produce CSV/plots/summaries; benchmark helper
streaming/     Holds sample.mp4; used by RTSP feeder
results/       Logs, CSVs, plots, summaries, demo, benchmarks (generated)
docker-compose.yml
.env
README.md
```

---

## .env Template

Place at project root:

```env
# Networking
CLOUD_HOST=cloud
CLOUD_PORT=5000

# Input: file | rtsp | webcam
INPUT_SOURCE=file
VIDEO_PATH=/app/streaming/sample.mp4
RTSP_URL=rtsp://mediamtx:8554/live.sdp

# Edge controls
EDGE_MAX_FPS=5
PERSON_CONF=0.4
SOURCE_ID=cam01
YOLO_MODEL=yolov8n.pt

# Benchmark toggles
EDGE_ACTIVITY=false
OFFLOAD_TO_CLOUD=true

# Cloud behavior
SAVE_ANNOTATIONS=false
LOITER_WINDOW_S=60
LOITER_RADIUS=0.03
FALL_PERSIST_S=10
NIGHT_START_HOUR=22
NIGHT_END_HOUR=6
```

---

## Build & Run (File Input, Edge+Cloud)

```bash
docker compose build
docker compose up -d cloud edge
curl -s http://localhost:5001/health
docker compose logs -f edge
```

Generated artifacts (after \~1–2 min):

* `results/metrics_edge.jsonl`
* `results/metrics_cloud.jsonl`

---

## Optional: RTSP Simulator (IP Camera Emulation)

```bash
docker compose up -d mediamtx feeder
```

Switch `.env` (if you want RTSP):

```
INPUT_SOURCE=rtsp
RTSP_URL=rtsp://mediamtx:8554/live.sdp
```

Then:

```bash
docker compose up -d edge
```

Play from host (optional): `rtsp://localhost:8554/live.sdp` (e.g., VLC).

---

## Monitoring & Analysis

Generate CSVs, plots, and summaries from logs:

```bash
docker compose build monitoring
docker compose run --rm monitoring \
  python analyze_logs.py /data/metrics_edge.jsonl /data/metrics_cloud.jsonl --out /data --plots

cat results/summary.txt
```

You’ll get:

* `results/edge_metrics.csv`, `results/cloud_metrics.csv`
* `results/summary.txt`, `results/summary.json`
* `results/edge_fps_ema.png`, `results/edge_infer_ms.png`, `results/edge_cloud_rtt_ms.png`

`summary.txt` includes p50/p95 latencies and an **end-to-end** approximation based on edge inference + edge-measured cloud RTT.

---

## Benchmark: Edge-Only vs Edge+Cloud

**Edge+Cloud (6 min):**

```bash
: > results/metrics_edge.jsonl
: > results/metrics_cloud.jsonl
sed -i '' 's/^EDGE_ACTIVITY=.*/EDGE_ACTIVITY=false/' .env || echo "EDGE_ACTIVITY=false" >> .env
sed -i '' 's/^OFFLOAD_TO_CLOUD=.*/OFFLOAD_TO_CLOUD=true/' .env || echo "OFFLOAD_TO_CLOUD=true" >> .env
docker compose up -d cloud edge
sleep 360
cp results/metrics_edge.jsonl  results/metrics_edge_edgecloud.jsonl
cp results/metrics_cloud.jsonl results/metrics_cloud_edgecloud.jsonl
```

**Edge-Only (6 min):**

```bash
: > results/metrics_edge.jsonl
sed -i '' 's/^EDGE_ACTIVITY=.*/EDGE_ACTIVITY=true/' .env || echo "EDGE_ACTIVITY=true" >> .env
sed -i '' 's/^OFFLOAD_TO_CLOUD=.*/OFFLOAD_TO_CLOUD=false/' .env || echo "OFFLOAD_TO_CLOUD=false" >> .env
docker compose up -d edge
sleep 360
cp results/metrics_edge.jsonl results/metrics_edge_edgeonly.jsonl
```

**Analyze both runs:**

```bash
mkdir -p results/edgecloud results/edgeonly
docker compose run --rm monitoring \
  python analyze_logs.py /data/metrics_edge_edgecloud.jsonl /data/metrics_cloud_edgecloud.jsonl --out /data/edgecloud --plots
docker compose run --rm monitoring \
  python analyze_logs.py /data/metrics_edge_edgeonly.jsonl /data/metrics_cloud_edgecloud.jsonl --out /data/edgeonly --plots
```

**Generate comparison table:**

```bash
docker compose run --rm monitoring python - <<'PY'
import json, statistics as st, os
A=json.load(open('/data/edgecloud/summary.json'))
B=json.load(open('/data/edgeonly/summary.json'))
edgeonly=[]
with open('/data/metrics_edge_edgeonly.jsonl','r',encoding='utf-8') as f:
    for line in f:
        try:
            d=json.loads(line)
            if d.get('stage')=='edge':
                im=d.get('infer_ms'); pm=d.get('pose_ms')
                if im is not None and pm is not None:
                    edgeonly.append(im+pm)
        except: pass
e2e_edgeonly_p50=round(st.median(edgeonly),1) if edgeonly else None
e2e_edgeonly_p95=round(sorted(edgeonly)[int(0.95*(len(edgeonly)-1))],1) if edgeonly else None
rows=[
 ("Edge inference p50 (ms)", A.get("edge",{}).get("infer_ms_p50"), B.get("edge",{}).get("infer_ms_p50")),
 ("Edge inference p95 (ms)", A.get("edge",{}).get("infer_ms_p95"), B.get("edge",{}).get("infer_ms_p95")),
 ("End-to-End p50 (ms)", A.get("e2e",{}).get("ms_p50"), e2e_edgeonly_p50),
 ("End-to-End p95 (ms)", A.get("e2e",{}).get("ms_p95"), e2e_edgeonly_p95),
 ("FPS (EMA mean)", A.get("edge",{}).get("fps_ema_mean"), B.get("edge",{}).get("fps_ema_mean")),
 ("CPU mean (%)", A.get("edge",{}).get("cpu_mean"), B.get("edge",{}).get("cpu_mean")),
 ("MEM mean (%)", A.get("edge",{}).get("mem_mean"), B.get("edge",{}).get("mem_mean")),
]
out=["| Metric | Edge+Cloud | Edge-Only |","|---|---:|---:|"]
for name,ac,eo in rows:
    out.append(f"| {name} | {ac if ac is not None else '-'} | {eo if eo is not None else '-'} |")
open('/data/benchmark_summary.md','w').write("\n".join(out)+"\n")
PY

cat results/benchmark_summary.md
```