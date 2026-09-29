"""INT8 static post-training quantization with ONNX Runtime.

Calibration uses windows from the *training* subjects only, so the test
subjects stay unseen by every step of the pipeline.
"""
import numpy as np
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                      quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

from data import MODELS_DIR, load_normalizer, load_split, normalize

N_CALIBRATION = 512


class IMUCalibrationReader(CalibrationDataReader):
    def __init__(self, X):
        self._iter = iter([{"imu": x[None]} for x in X])

    def get_next(self):
        return next(self._iter, None)


def main():
    X_train, y_train, _ = load_split("train")
    X_train = normalize(X_train, *load_normalizer())

    # Class-balanced calibration sample from the training set
    rng = np.random.default_rng(42)
    per_class = N_CALIBRATION // 6
    idx = np.concatenate([rng.choice(np.flatnonzero(y_train == c), per_class, replace=False)
                          for c in range(6)])

    fp32_path = MODELS_DIR / "har_fp32.onnx"
    prep_path = MODELS_DIR / "har_fp32_prep.onnx"
    int8_path = MODELS_DIR / "har_int8.onnx"

    # Shape inference + graph optimisation (folds BatchNorm into Conv)
    quant_pre_process(str(fp32_path), str(prep_path))
    quantize_static(
        str(prep_path), str(int8_path), IMUCalibrationReader(X_train[idx]),
        quant_format=QuantFormat.QOperator,  # QLinearConv/QLinearMatMul: smaller than QDQ
        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
        per_channel=True,
    )
    prep_path.unlink()

    fp32_kb, int8_kb = fp32_path.stat().st_size / 1024, int8_path.stat().st_size / 1024
    print(f"FP32: {fp32_kb:.1f} KB -> INT8: {int8_kb:.1f} KB  ({fp32_kb / int8_kb:.2f}x smaller)")


if __name__ == "__main__":
    main()
