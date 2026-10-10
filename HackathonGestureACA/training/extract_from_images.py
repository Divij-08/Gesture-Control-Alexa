"""Extract fingerspelling training data from a folder of RGB letter images.

Bridges a downloaded image dataset (e.g. Kaggle "ASL Alphabet", grassknoted)
into the SAME landmark CSV our webcam collector produces, so no camera session
is needed. Runs the shared capture/landmarks.py extractor over each image and
appends normalized 63-feature rows to data/fingerspell.csv (64 cols: 63 + label).

Because it reuses capture.landmarks.extract_landmarks, the feature space is
identical to live inference — the train==test guarantee (design §21.0) holds.

Expected layout: one subfolder per class, folder name = the letter label:
    <root>/A/*.jpg
    <root>/B/*.jpg
    ...
Single-letter folder names only; anything else (del, nothing, space, ...) is
skipped. J and Z are skipped (motion letters — design §15), matching the
webcam collector.

USAGE:
    python -m training.extract_from_images <root> [--per-class N] [--min-conf F]

    <root>        dir containing the per-letter subfolders
    --per-class N cap images processed per letter (default 600; datasets often
                  ship 3000+/letter which is slow and redundant)
    --min-conf F  MediaPipe min_detection_confidence (default 0.5; stills are
                  harder than tracked video so we go a touch lower than live)

Images where MediaPipe finds no hand are counted and skipped (reported at end).
Run training.train_fingerspell afterwards exactly as with webcam data.
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import Counter

import cv2

from capture.landmarks import make_hands, extract_landmarks, NUM_FEATURES

DATA_PATH = os.path.join("data", "fingerspell.csv")
SKIP = {"j", "z"}  # motion letters — keep in lockstep with training/collect_data.py
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_class_dirs(root: str):
    """Yield (label, dirpath) for each single-letter subfolder, excluding SKIP."""
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        label = name.strip().lower()
        if len(label) != 1 or not label.isalpha():
            print(f"  skip folder '{name}' (not a single letter)")
            continue
        if label in SKIP:
            print(f"  skip '{label}' (motion letter, design §15)")
            continue
        yield label, path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="dir containing per-letter subfolders")
    ap.add_argument("--per-class", type=int, default=600)
    ap.add_argument("--min-conf", type=float, default=0.5)
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        raise SystemExit(f"Not a directory: {args.root}")

    os.makedirs("data", exist_ok=True)
    # static_image_mode=True: treat each file as an independent still, not a video frame.
    hands = make_hands(max_num_hands=1, min_conf=args.min_conf, static_image_mode=True)

    written = Counter()   # rows written per letter
    no_hand = Counter()   # images where MediaPipe found no hand
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
                feats = extract_landmarks(hands.process(rgb))
                if feats is None or len(feats) != NUM_FEATURES:
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
        print("WARNING: 0 rows. Check the folder layout (one letter-named subfolder "
              "per class) and that images are large enough for hand detection.")
    print("\nNext: python -m training.train_fingerspell")


if __name__ == "__main__":
    main()
