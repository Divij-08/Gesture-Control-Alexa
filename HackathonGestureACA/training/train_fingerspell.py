"""Train the fingerspelling MLP on collected landmark data.

Reads data/fingerspell.csv, trains a scikit-learn MLPClassifier, reports
accuracy + a confusion matrix (so you can spot look-alike letters), and saves
the model to recognizer/fingerspell/model.joblib. See design doc §21.1.

USAGE:
    python -m training.train_fingerspell

Why scikit-learn MLP (not TensorFlow): this is a tiny static per-frame classifier;
sklearn trains it in seconds, saves in one line, and avoids TF install pain.
It is a genuine multi-layer perceptron, matching the design.
"""
from __future__ import annotations

import csv
import os

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

from capture.landmarks import NUM_FEATURES, mirror_landmarks

DATA_PATH = os.path.join("data", "fingerspell.csv")
MODEL_PATH = os.path.join("recognizer", "fingerspell", "model.joblib")
MIN_PER_CLASS = 30  # warn below this
AUGMENT_MIRROR = True  # add the left/right-hand mirror of each TRAIN sample (handedness-invariant)

# Letters to drop from training. 'S' (a plain fist) is too easily confused live with
# 'A' (a fist with the thumb alongside) — excluding it removes that misfire. The CSV
# is left intact; this just filters at load time, so it's reversible by editing here.
EXCLUDE = {"s"}


def load_data() -> tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(DATA_PATH):
        raise SystemExit(f"No data at {DATA_PATH}. Run: python -m training.collect_data")
    X, y = [], []
    skipped = 0
    with open(DATA_PATH, newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            if len(row) != NUM_FEATURES + 1:
                continue  # skip malformed rows
            label = row[-1].strip().lower()
            if label in EXCLUDE:
                skipped += 1
                continue
            X.append([float(v) for v in row[:NUM_FEATURES]])
            y.append(label)
    if EXCLUDE:
        print(f"Excluding {sorted(EXCLUDE)} ({skipped} rows dropped).")
    return np.array(X, dtype=np.float32), np.array(y)


def main() -> None:
    X, y = load_data()
    classes, counts = np.unique(y, return_counts=True)
    print(f"Loaded {len(X)} samples across {len(classes)} letters.")
    for c, n in zip(classes, counts):
        flag = "  <-- LOW, collect more" if n < MIN_PER_CLASS else ""
        print(f"  {c.upper()}: {n}{flag}")

    if len(classes) < 2:
        raise SystemExit("Need at least 2 letters to train. Collect more data.")

    # Stratified split keeps class balance in train/test (design §21.1 step 3).
    # NOTE: we split the RAW data FIRST, then augment only the training half.
    # If we mirrored before splitting, a sample and its mirror could land on
    # opposite sides of the split and inflate the test score (leakage).
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Handedness-invariant augmentation: add the left/right mirror of each TRAIN
    # sample (negate x). Doubles the training set so the model recognizes a letter
    # signed with either hand. Test sets stay un-augmented so scores are honest.
    if AUGMENT_MIRROR:
        X_tr_m = np.array([mirror_landmarks(x) for x in X_tr], dtype=np.float32)
        X_tr = np.concatenate([X_tr, X_tr_m])
        y_tr = np.concatenate([y_tr, y_tr])
        print(f"\nMirror augmentation ON: train {len(X_tr)//2} -> {len(X_tr)} samples.")

    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        alpha=1e-4,             # L2 regularization
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
    )
    print("\nTraining MLP (128, 64)...")
    clf.fit(X_tr, y_tr)

    acc = clf.score(X_te, y_te)
    print(f"\n=== TEST ACCURACY (original hand): {acc:.1%} ===")

    # Prove handedness invariance: score the SAME held-out test set mirrored to the
    # other hand. With augmentation on, this should be ~equal to the original-hand
    # score; without it, expect a large drop (that's the whole point of augmenting).
    X_te_m = np.array([mirror_landmarks(x) for x in X_te], dtype=np.float32)
    acc_m = clf.score(X_te_m, y_te)
    print(f"=== TEST ACCURACY (mirrored/other hand): {acc_m:.1%} ===\n")

    y_pred = clf.predict(X_te)
    print(classification_report(y_te, y_pred, zero_division=0))

    # Confusion matrix — look for off-diagonal mass on look-alike letters
    # (M/N/S, D/R/U, T/K). See design §21.1 step 5 (confusable-set trick).
    print("Confusion matrix (rows=true, cols=pred):")
    labels = sorted(classes)
    cm = confusion_matrix(y_te, y_pred, labels=labels)
    header = "     " + " ".join(f"{l.upper():>3}" for l in labels)
    print(header)
    for l, row in zip(labels, cm):
        print(f"  {l.upper():>2} " + " ".join(f"{v:>3}" for v in row))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": clf, "labels": labels}, MODEL_PATH)
    print(f"\nsaved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
