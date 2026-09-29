"""Export the trained PyTorch CNN to ONNX and check it matches PyTorch."""
import numpy as np
import onnx
import onnxruntime as ort
import torch

from data import MODELS_DIR, load_normalizer, load_split, normalize
from model import HARCNN


def main():
    model = HARCNN()
    model.load_state_dict(torch.load(MODELS_DIR / "har_fp32.pt", map_location="cpu"))
    model.eval()

    onnx_path = MODELS_DIR / "har_fp32.onnx"
    torch.onnx.export(
        model, torch.randn(1, 9, 128), onnx_path,
        input_names=["imu"], output_names=["logits"],
        dynamic_axes={"imu": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    onnx.checker.check_model(onnx.load(onnx_path))

    # Numerical parity check on real test windows
    X_test, _, _ = load_split("test")
    X = normalize(X_test[:256], *load_normalizer())
    with torch.no_grad():
        torch_out = model(torch.from_numpy(X)).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"imu": X})[0]

    max_diff = np.abs(torch_out - ort_out).max()
    assert np.allclose(torch_out, ort_out, atol=1e-4), f"max abs diff {max_diff}"
    assert (torch_out.argmax(1) == ort_out.argmax(1)).all()
    print(f"Exported {onnx_path} ({onnx_path.stat().st_size / 1024:.1f} KB); "
          f"PyTorch vs ONNX Runtime max abs diff = {max_diff:.2e}")


if __name__ == "__main__":
    main()
