"""Streamlit demo for the INT8 ONNX activity classifier.

Input: a CSV where each row is one raw IMU window flattened to 1152 values
(9 channels x 128 samples, channel-major in the order of src/data.py SIGNALS),
with an optional 'Activity' column for evaluation. Without an upload, demo
windows from the UCI test subjects are used (requires data/UCI HAR Dataset).

Run:  streamlit run app.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report

sys.path.insert(0, str(Path(__file__).parent / "src"))
from data import DATA_DIR, LABELS, MODELS_DIR, load_normalizer, load_split, normalize  # noqa: E402

N_CHANNELS, WINDOW = 9, 128


@st.cache_resource
def load_model(name):
    return ort.InferenceSession(str(MODELS_DIR / f"har_{name}.onnx"), providers=["CPUExecutionProvider"])


@st.cache_data
def demo_windows(n=500, seed=0):
    X, y, _ = load_split("test")
    idx = np.random.default_rng(seed).choice(len(X), n, replace=False)
    return X[idx], np.array(LABELS)[y[idx]]


def parse_csv(file):
    df = pd.read_csv(file)
    labels = df.pop("Activity").values if "Activity" in df.columns else None
    if df.shape[1] != N_CHANNELS * WINDOW:
        st.error(f"Expected {N_CHANNELS * WINDOW} signal columns per row, got {df.shape[1]}.")
        st.stop()
    return df.values.astype(np.float32).reshape(-1, N_CHANNELS, WINDOW), labels


st.title("Human Activity Recognition — Edge Model")
st.caption("1-D CNN on raw 9-axis IMU windows, INT8-quantized ONNX Runtime model")

model_name = st.sidebar.radio("Model", ["int8", "fp32"], format_func=str.upper)
uploaded = st.sidebar.file_uploader("Upload raw IMU windows (CSV)", type="csv")

if uploaded is not None:
    X, y_true = parse_csv(uploaded)
elif DATA_DIR.exists():
    st.info("No file uploaded — using 500 random windows from the UCI test subjects.")
    X, y_true = demo_windows()
else:
    st.warning("Upload a CSV, or download the UCI HAR dataset into data/ for demo windows.")
    st.stop()

# Normalize with the statistics saved at training time (never refit on new data)
x = normalize(X, *load_normalizer())
logits = load_model(model_name).run(None, {"imu": x})[0]
probs = np.exp(logits - logits.max(1, keepdims=True))
probs /= probs.sum(1, keepdims=True)
y_pred = np.array(LABELS)[probs.argmax(1)]

results = pd.DataFrame({"prediction": y_pred, "confidence": probs.max(1).round(3)})
if y_true is not None:
    results.insert(0, "true", y_true)

col1, col2 = st.columns(2)
col1.metric("Windows", len(X))
col2.metric("Mean confidence", f"{results['confidence'].mean():.2f}")
st.dataframe(results, use_container_width=True, height=250)

if y_true is not None:
    st.subheader(f"Accuracy: {accuracy_score(y_true, y_pred):.2%}")
    report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose().round(3))
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=LABELS, cmap=plt.cm.Blues,
                                            ax=ax, xticks_rotation=45, colorbar=False)
    st.pyplot(fig)

st.subheader("Inspect a window")
i = st.slider("Window index", 0, len(X) - 1, 0)
fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
for ax, (name, sl) in zip(axes, [("body acc", slice(0, 3)), ("gyro", slice(3, 6)), ("total acc", slice(6, 9))]):
    ax.plot(X[i, sl].T)
    ax.set_ylabel(name)
axes[0].set_title(f"Predicted: {y_pred[i]} ({probs[i].max():.2f})"
                  + (f" — true: {y_true[i]}" if y_true is not None else ""))
axes[-1].set_xlabel("sample (50 Hz)")
st.pyplot(fig)
