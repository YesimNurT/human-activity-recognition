"""Leakage and model-parity checks (require data/UCI HAR Dataset and trained models)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import DATA_DIR, MODELS_DIR, fit_normalizer, load_normalizer, load_split, normalize  # noqa: E402

needs_data = pytest.mark.skipif(not DATA_DIR.exists(), reason="UCI HAR dataset not downloaded")
needs_models = pytest.mark.skipif(not (MODELS_DIR / "har_int8.onnx").exists(), reason="models not built")


@needs_data
def test_train_test_subjects_disjoint():
    _, _, s_train = load_split("train")
    _, _, s_test = load_split("test")
    assert set(s_train).isdisjoint(s_test)


@needs_data
def test_cv_folds_are_subject_disjoint():
    from sklearn.model_selection import StratifiedGroupKFold
    X, y, s = load_split("train")
    for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y, groups=s):
        assert set(s[tr]).isdisjoint(s[va])


@needs_data
@needs_models
def test_saved_normalizer_matches_training_data():
    X, _, _ = load_split("train")
    for saved, fresh in zip(load_normalizer(), fit_normalizer(X)):
        np.testing.assert_allclose(saved, fresh, rtol=1e-5)


@needs_data
@needs_models
def test_int8_agrees_with_fp32():
    import onnxruntime as ort
    X, _, _ = load_split("test")
    x = normalize(X[:500], *load_normalizer())
    preds = [ort.InferenceSession(str(MODELS_DIR / f"har_{n}.onnx")).run(None, {"imu": x})[0].argmax(1)
             for n in ("fp32", "int8")]
    assert (preds[0] == preds[1]).mean() > 0.97
