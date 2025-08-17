import os, time, json, psutil, requests

CLOUD_HOST = os.getenv("CLOUD_HOST", "cloud")
CLOUD_PORT = int(os.getenv("CLOUD_PORT", "5000"))
INPUT_SOURCE = os.getenv("INPUT_SOURCE", "webcam")

def log(msg: str, **kv):
    rec = {"ts": time.time(), "msg": msg, **kv}
    print(f"[EDGE] {json.dumps(rec)}", flush=True)

def ping_cloud():
    url = f"http://{CLOUD_HOST}:{CLOUD_PORT}/health"
    try:
        r = requests.get(url, timeout=3)
        body = r.json() if r.ok else {}
        log("cloud_health", ok=r.ok, status=r.status_code, body=body)
    except Exception as e:
        log("cloud_health_error", err=str(e))

def main():
    log("boot", input_source=INPUT_SOURCE)
    ping_cloud()
    start = time.time()
    while True:
        cpu = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory().percent
        log("heartbeat", cpu=cpu, mem=mem)
        if time.time() - start > 10:
            log("stub_note", note="Replace stub with real detection in Step 3")
            start = time.time()

if __name__ == "__main__":
    main()
