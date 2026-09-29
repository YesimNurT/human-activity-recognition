"""Leak-free training of the raw-IMU CNN.

1. 5-fold StratifiedGroupKFold over training *subjects*. Inside each fold a
   further subject-wise split provides the early-stopping set, so the outer
   validation subjects are only used for scoring.
2. Normalization statistics are fitted on training windows only.
3. The final model is trained on all training subjects for the median best
   epoch from CV and evaluated once on the official test subjects.

Usage:  python src/train.py            (full CV + final model)
        python src/train.py --skip-cv  (final model only)
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from torch.utils.data import DataLoader, TensorDataset

from data import LABELS, MODELS_DIR, fit_normalizer, load_split, normalize, save_normalizer
from model import HARCNN

SEED = 42
BATCH_SIZE = 64
LR = 1e-3
MAX_EPOCHS = 40
PATIENCE = 6


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(X, y, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad()
        criterion(model(xb), yb).backward()
        optimizer.step()


@torch.no_grad()
def predict(model, X):
    model.eval()
    return torch.cat([model(xb) for (xb,) in DataLoader(TensorDataset(torch.from_numpy(X)),
                                                          batch_size=512)]).numpy()


def fit_with_early_stopping(X_tr, y_tr, X_es, y_es):
    """Train until the early-stopping loss stops improving; return best model and epoch."""
    set_seed()
    model = HARCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loader = make_loader(X_tr, y_tr, shuffle=True)

    best_loss, best_epoch, best_state, bad = float("inf"), 0, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        train_epoch(model, loader, criterion, optimizer)
        es_loss = criterion(torch.from_numpy(predict(model, X_es)), torch.from_numpy(y_es)).item()
        if es_loss < best_loss:
            best_loss, best_epoch, bad = es_loss, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best_epoch


def fit_fixed_epochs(X, y, n_epochs):
    set_seed()
    model = HARCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loader = make_loader(X, y, shuffle=True)
    for _ in range(n_epochs):
        train_epoch(model, loader, criterion, optimizer)
    return model.eval()


def cross_validate(X, y, subjects, n_splits=5):
    outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    fold_acc, fold_f1, best_epochs = [], [], []

    for fold, (tr, va) in enumerate(outer.split(X, y, groups=subjects), start=1):
        # Inner subject-wise split for early stopping
        inner = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
        fit_i, es_i = next(inner.split(X[tr], y[tr], groups=subjects[tr]))
        fit_idx, es_idx = tr[fit_i], tr[es_i]
        assert not set(subjects[fit_idx]) & set(subjects[es_idx])
        assert not set(subjects[tr]) & set(subjects[va])

        mean, std = fit_normalizer(X[fit_idx])
        t0 = time.time()
        model, best_epoch = fit_with_early_stopping(
            normalize(X[fit_idx], mean, std), y[fit_idx],
            normalize(X[es_idx], mean, std), y[es_idx])
        pred = predict(model, normalize(X[va], mean, std)).argmax(1)

        acc, f1 = accuracy_score(y[va], pred), f1_score(y[va], pred, average="macro")
        fold_acc.append(acc)
        fold_f1.append(f1)
        best_epochs.append(best_epoch)
        print(f"Fold {fold}: val subjects {sorted(set(subjects[va]))} | best epoch {best_epoch:2d} "
              f"| acc {acc:.4f} | macro-F1 {f1:.4f} | {time.time() - t0:.0f}s")

    fold_acc, fold_f1 = np.array(fold_acc), np.array(fold_f1)
    print(f"\nSubject-wise CV accuracy : {fold_acc.mean():.4f} ± {fold_acc.std():.4f}")
    print(f"Subject-wise CV macro-F1 : {fold_f1.mean():.4f} ± {fold_f1.std():.4f}")
    return {
        "cv_accuracy_mean": float(fold_acc.mean()), "cv_accuracy_std": float(fold_acc.std()),
        "cv_macro_f1_mean": float(fold_f1.mean()), "cv_macro_f1_std": float(fold_f1.std()),
        "cv_fold_accuracy": fold_acc.round(4).tolist(), "cv_best_epochs": best_epochs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--epochs", type=int, default=None,
                        help="final-model epochs (default: median best epoch from CV)")
    args = parser.parse_args()

    X_train, y_train, s_train = load_split("train")
    X_test, y_test, s_test = load_split("test")
    assert not set(s_train) & set(s_test), "train/test subjects overlap"
    print(f"Train: {X_train.shape} from {len(set(s_train))} subjects | "
          f"Test: {X_test.shape} from {len(set(s_test))} subjects")

    metrics = {}
    if not args.skip_cv:
        metrics.update(cross_validate(X_train, y_train, s_train))

    n_epochs = args.epochs or int(np.median(metrics.get("cv_best_epochs", [20])))
    print(f"\nTraining final model on all training subjects for {n_epochs} epochs ...")
    mean, std = fit_normalizer(X_train)
    model = fit_fixed_epochs(normalize(X_train, mean, std), y_train, n_epochs)

    pred = predict(model, normalize(X_test, mean, std)).argmax(1)
    acc, f1 = accuracy_score(y_test, pred), f1_score(y_test, pred, average="macro")
    print(f"\nTest accuracy (9 unseen subjects): {acc:.4f} | macro-F1: {f1:.4f}\n")
    print(classification_report(y_test, pred, target_names=LABELS, digits=4))
    print(confusion_matrix(y_test, pred))

    MODELS_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / "har_fp32.pt")
    save_normalizer(mean, std)
    metrics.update({"final_epochs": n_epochs, "test_accuracy": float(acc), "test_macro_f1": float(f1),
                    "n_params": sum(p.numel() for p in model.parameters())})
    (MODELS_DIR / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved {MODELS_DIR / 'har_fp32.pt'}, norm.npz, train_metrics.json")


if __name__ == "__main__":
    main()
