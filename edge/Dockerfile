FROM python:3.10-slim

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 v4l-utils \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY edge_main.py .

CMD ["python", "edge_main.py"]
