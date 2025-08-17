from flask import Flask, jsonify, request

app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify({"ok": True, "service": "cloud", "msg": "cloud up"}), 200

@app.post("/analyze")
def analyze():
    # Accept image later; for now just return a dummy activity
    return jsonify({"activity": "unknown_stub", "alert": None}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
