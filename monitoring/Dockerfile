FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY analyze_logs.py .
CMD ["python", "analyze_logs.py", "/data/metrics_edge.jsonl", "/data/metrics_cloud.jsonl", "--out", "/data", "--plots"]
