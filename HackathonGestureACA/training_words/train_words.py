"""Train the word-level sign MLP on collected landmark data.

Reads landmark CSV(s), trains a scikit-learn MLPClassifier, reports accuracy +
a confusion matrix, and saves the model to recognizer/words/model.joblib.

USAGE:
    python -m training_words.train_words                                # data/words.csv, all classes
    python -m training_words.train_words --data data/words_webcam.csv   # webcam-collected data
    python -m training_words.train_words --incar-only                   # only the 40 in-car words
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

from capture.landmarks import NUM_FEATURES_BOTH, mirror_landmarks_both
from training_words.words_list import INCAR_WORDS

DEFAULT_DATA_PATH = os.path.join("data", "words.csv")
MODEL_PATH = os.path.join("recognizer", "words", "model.joblib")
MIN_PER_CLASS = 30
AUGMENT_MIRROR = True


def load_data(data_path: str, classes: set[str] | None) -> tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(data_path):
        raise SystemExit(
            f"No data at {data_path}. Run: python -m training_words.collect_data "
            f"or python -m training_words.extract_from_images <root>"
        )
    X, y = [], []
    with open(data_path, newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            if len(row) != NUM_FEATURES_BOTH + 1:
                continue
            label = row[-1]
            if classes is not None and label not in classes:
                continue
            X.append([float(v) for v in row[:NUM_FEATURES_BOTH]])
            y.append(label)
    return np.array(X, dtype=np.float32), np.array(y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA_PATH,
                    help=f"landmark CSV to train on (default {DEFAULT_DATA_PATH})")
    ap.add_argument("--incar-only", action="store_true",
                    help="train only on the 40 in-car words (words_list.py)")
    args = ap.parse_args()

    classes_filter = set(INCAR_WORDS) if args.incar_only else None
    X, y = load_data(args.data, classes_filter)
    if len(X) == 0:
        raise SystemExit("No usable rows found (check --data path / --incar-only filter).")
    classes, counts = np.unique(y, return_counts=True)
    print(f"Loaded {len(X)} samples across {len(classes)} words.")
    for c, n in zip(classes, counts):
        flag = "  <-- LOW, collect more" if n < MIN_PER_CLASS else ""
        print(f"  {c.upper()}: {n}{flag}")

    if len(classes) < 2:
        raise SystemExit("Need at least 2 word classes to train. Collect more data.")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if AUGMENT_MIRROR:
        X_tr_m = np.array([mirror_landmarks_both(x) for x in X_tr], dtype=np.float32)
        X_tr = np.concatenate([X_tr, X_tr_m])
        y_tr = np.concatenate([y_tr, y_tr])
        print(f"\nMirror augmentation ON: train {len(X_tr)//2} -> {len(X_tr)} samples.")

    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        alpha=1e-4,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
    )
    print("\nTraining MLP (256, 128)...")
    clf.fit(X_tr, y_tr)

    acc = clf.score(X_te, y_te)
    print(f"\n=== TEST ACCURACY (original hand): {acc:.1%} ===")

    X_te_m = np.array([mirror_landmarks_both(x) for x in X_te], dtype=np.float32)
    acc_m = clf.score(X_te_m, y_te)
    print(f"=== TEST ACCURACY (mirrored/other hand): {acc_m:.1%} ===\n")

    y_pred = clf.predict(X_te)
    print(classification_report(y_te, y_pred, zero_division=0))

    print("Confusion matrix (rows=true, cols=pred):")
    labels = sorted(classes)
    cm = confusion_matrix(y_te, y_pred, labels=labels)
    header = "     " + " ".join(f"{l[:6]:>6}" for l in labels)
    print(header)
    for l, row in zip(labels, cm):
        print(f"  {l[:6]:>6} " + " ".join(f"{v:>6}" for v in row))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": clf, "labels": labels}, MODEL_PATH)
    print(f"\nsaved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
