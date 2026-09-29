"""Compare FP32 vs INT8 ONNX models: size, single-window latency, test accuracy."""
import json
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
from sklearn.metrics import accuracy_score, f1_score

from data import MODELS_DIR, load_normalizer, load_split, normalize

WARMUP, RUNS = 50, 1000


def latency_ms(sess, x):
    for _ in range(WARMUP):
        sess.run(None, {"imu": x})
    t0 = time.perf_counter()
    for _ in range(RUNS):
        sess.run(None, {"imu": x})
    return (time.perf_counter() - t0) / RUNS * 1000


def weight_kb(path):
    """Bytes held by learned weight tensors (excludes graph structure and quantization scales)."""
    model = onnx.load(path)
    return sum(numpy_helper.to_array(t).nbytes for t in model.graph.initializer
               if len(t.dims) > 1) / 1024


def main():
    X_test, y_test, _ = load_split("test")
    X_test = normalize(X_test, *load_normalizer())

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1  # single core, closer to an edge device

    results, preds = {}, {}
    for name in ["fp32", "int8"]:
        path = MODELS_DIR / f"har_{name}.onnx"
        sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        pred = sess.run(None, {"imu": X_test})[0].argmax(1)
        preds[name] = pred
        results[name] = {
            "size_kb": round(path.stat().st_size / 1024, 1),
            "weights_kb": round(weight_kb(path), 1),
            "latency_ms": round(latency_ms(sess, X_test[:1]), 3),
            "test_accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "test_macro_f1": round(float(f1_score(y_test, pred, average="macro")), 4),
        }

    results["size_reduction_x"] = round(results["fp32"]["size_kb"] / results["int8"]["size_kb"], 2)
    results["weight_reduction_x"] = round(results["fp32"]["weights_kb"] / results["int8"]["weights_kb"], 2)
    results["prediction_agreement"] = round(float((preds["fp32"] == preds["int8"]).mean()), 4)
    (MODELS_DIR / "benchmark.json").write_text(json.dumps(results, indent=2))

    print("| Model | File size (KB) | Weights (KB) | Latency (ms/window, 1 thread) | Test accuracy | Test macro-F1 |")
    print("|---|---|---|---|---|---|")
    for name in ["fp32", "int8"]:
        r = results[name]
        print(f"| {name.upper()} ONNX | {r['size_kb']} | {r['weights_kb']} | {r['latency_ms']} | "
              f"{r['test_accuracy']:.4f} | {r['test_macro_f1']:.4f} |")
    print(f"\nFile size reduction: {results['size_reduction_x']}x | "
          f"weight reduction: {results['weight_reduction_x']}x | "
          f"FP32/INT8 prediction agreement: {results['prediction_agreement']:.2%}")


if __name__ == "__main__":
    main()
