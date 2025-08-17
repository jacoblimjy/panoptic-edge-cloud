# Real-Time Edge–Cloud Person Activity Recognition

Scaffolded two-container app:
- **edge/**: will run real-time person detection, sampling, offload
- **cloud/**: will run activity recognition & contextual alerts
- **streaming/**: sample media / RTSP configs
- **monitoring/**: log analysis
- **results/**: logs & artifacts

## Quick start
```bash
docker compose build
docker compose up


