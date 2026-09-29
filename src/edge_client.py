"""Edge inference client: INT8 ONNX model on-device, predictions sent to AWS.

Replays windows from the UCI test subjects as a simulated live sensor stream,
classifies each window locally with ONNX Runtime and posts the predictions in
batches to the monitoring API. Network failures never block inference:
unsent predictions stay buffered and are retried with the next batch.

Configuration (environment or a .env file in the repo root):
    HAR_API_URL   e.g. https://abc123.execute-api.eu-central-1.amazonaws.com/prod/predictions
    HAR_API_KEY   API Gateway key

Usage:
    python src/edge_client.py --dry-run            # no network, print only
    python src/edge_client.py --n 300 --rate 20    # 300 windows at 20 windows/s
"""
import argparse
import os
import time
import uuid

import numpy as np
import onnxruntime as ort
import requests
from dotenv import load_dotenv

from data import LABELS, MODELS_DIR, ROOT, load_normalizer, load_split, normalize

MODEL_VERSION = "har-cnn-int8-v1"
MAX_BUFFER = 1000


class EdgeClassifier:
    def __init__(self, model_path=MODELS_DIR / "har_int8.onnx"):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self.mean, self.std = load_normalizer()

    def predict(self, window):
        """window: raw (9, 128) IMU window -> (label, confidence, latency_ms)."""
        x = normalize(window[None], self.mean, self.std)
        t0 = time.perf_counter()
        logits = self.session.run(None, {"imu": x})[0][0]
        latency_ms = (time.perf_counter() - t0) * 1000
        p = np.exp(logits - logits.max())
        p /= p.sum()
        return LABELS[int(p.argmax())], float(p.max()), latency_ms


class MonitoringSender:
    def __init__(self, url, api_key, batch_size):
        self.url, self.batch_size = url, batch_size
        self.headers = {"x-api-key": api_key, "Content-Type": "application/json"}
        self.buffer = []

    def add(self, record):
        self.buffer.append(record)
        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        batch = self.buffer[:100]  # API accepts up to 100 records per request
        try:
            r = requests.post(self.url, json={"predictions": batch}, headers=self.headers, timeout=5)
            r.raise_for_status()
            del self.buffer[:len(batch)]
            print(f"  -> sent {len(batch)} predictions ({r.status_code})")
        except requests.RequestException as e:
            print(f"  !! send failed, keeping {len(self.buffer)} buffered: {e}")
            del self.buffer[:max(0, len(self.buffer) - MAX_BUFFER)]  # bound memory on long outages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="number of windows to replay")
    parser.add_argument("--rate", type=float, default=10.0, help="windows per second (0 = no delay)")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--device-id", default=f"edge-{uuid.getnode() % 10000:04d}")
    parser.add_argument("--dry-run", action="store_true", help="classify only, do not send")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    sender = None
    if not args.dry_run:
        url, key = os.environ.get("HAR_API_URL"), os.environ.get("HAR_API_KEY")
        if not url or not key:
            raise SystemExit("Set HAR_API_URL and HAR_API_KEY (or use --dry-run).")
        sender = MonitoringSender(url, key, args.batch_size)

    clf = EdgeClassifier()
    X, y, subjects = load_split("test")
    rng = np.random.default_rng()
    start = rng.integers(0, len(X) - args.n)  # contiguous slice = realistic activity sequence

    correct = 0
    for i in range(start, start + args.n):
        label, conf, latency = clf.predict(X[i])
        correct += label == LABELS[y[i]]
        print(f"[{i}] subject {subjects[i]:2d} | pred {label:<18} conf {conf:.2f} | "
              f"true {LABELS[y[i]]:<18} | {latency:.2f} ms")
        if sender:
            sender.add({
                "device_id": args.device_id,
                "ts": int(time.time() * 1000),
                "label": label,
                "confidence": round(conf, 4),
                "latency_ms": round(latency, 3),
                "model_version": MODEL_VERSION,
                "true_label": LABELS[y[i]],  # available only in replay mode
            })
        if args.rate > 0:
            time.sleep(1 / args.rate)

    if sender:
        sender.flush()
    print(f"\nReplay accuracy: {correct / args.n:.2%} over {args.n} windows")


if __name__ == "__main__":
    main()
