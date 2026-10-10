"""Extract word-level training data from a folder of images.

Reads a dataset structured as one subfolder per word (folder name = word label),
runs the shared capture/landmarks.py extractor over each image, and appends
normalized 63-feature rows to data/words.csv (64 cols: 63 features + label).

Because it reuses capture.landmarks.extract_landmarks, the feature space is
identical to live inference — the train==test guarantee holds.

Expected layout:
    <root>/hello/*.jpg
    <root>/navigate/*.jpg
    <root>/call/*.jpg
    ...

Any subfolder name that contains at least one alphabetic character is accepted
as a class label (lowercased). Hidden folders (starting with '.') are skipped.

USAGE:
    python -m training_words.extract_from_images <root> [--per-class N] [--min-conf F]

    <root>        dir containing the per-word subfolders
    --per-class N cap images processed per word (default 600)
    --min-conf F  MediaPipe min_detection_confidence (default 0.5)

Images where MediaPipe finds no hand are counted and skipped (reported at end).
Run training_words.train_words afterwards to train the model.
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import Counter

import cv2

from capture.landmarks import make_hands, extract_landmarks_both_hands, NUM_FEATURES_BOTH

DATA_PATH = os.path.join("data", "words.csv")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_class_dirs(root: str):
    """Yield (label, dirpath) for each valid subfolder."""
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        if name.startswith("."):
            continue
        label = name.strip().lower()
        if not label or not any(c.isalpha() for c in label):
            print(f"  skip folder '{name}' (no alphabetic characters)")
            continue
        yield label, path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="dir containing per-word subfolders")
    ap.add_argument("--per-class", type=int, default=600)
    ap.add_argument("--min-conf", type=float, default=0.5)
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        raise SystemExit(f"Not a directory: {args.root}")

    os.makedirs("data", exist_ok=True)
    hands = make_hands(max_num_hands=2, min_conf=args.min_conf, static_image_mode=True)

    written = Counter()
    no_hand = Counter()
    out = open(DATA_PATH, "a", newline="")
    writer = csv.writer(out)

    try:
        for label, dirpath in iter_class_dirs(args.root):
            files = [f for f in sorted(os.listdir(dirpath))
                     if os.path.splitext(f)[1].lower() in IMG_EXTS]
            files = files[: args.per_class]
            print(f"[{label.upper()}] {len(files)} images...", end="", flush=True)

            for fname in files:
                img = cv2.imread(os.path.join(dirpath, fname))
                if img is None:
                    continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                feats = extract_landmarks_both_hands(hands.process(rgb))
                if feats is None or len(feats) != NUM_FEATURES_BOTH:
                    no_hand[label] += 1
                    continue
                writer.writerow([*feats.tolist(), label])
                written[label] += 1

            print(f" kept {written[label]}, no-hand {no_hand[label]}")
    finally:
        out.close()
        hands.close()

    total = sum(written.values())
    print("\n=== extracted (appended) ===")
    for label in sorted(written):
        miss = no_hand[label]
        rate = miss / (written[label] + miss) if (written[label] + miss) else 0.0
        print(f"  {label.upper()}: {written[label]}  (no-hand {miss}, {rate:.0%} miss)")
    print(f"total rows added: {total}  ->  {DATA_PATH}")
    if total == 0:
        print("WARNING: 0 rows. Check the folder layout (one word-named subfolder "
              "per class) and that images are large enough for hand detection.")
    print("\nNext: python -m training_words.train_words")


if __name__ == "__main__":
    main()
