"""Collect word-level sign training data from the webcam — with reference images.

For each of the 40 in-car words (words_list.py), shows you HOW to sign it by
displaying reference photos of real signers (before/mid/after motion phases,
pulled from data/word_image_dataset_3frames/<word>/), side by side with your
live webcam view. You copy the sign, hold it, and record.

Records normalized two-hand landmark vectors (126 features) labeled with the
word, appended to data/words_webcam.csv (127 cols: 126 features + label).

USAGE:
    python -m training_words.collect_data

CONTROLS (focus the video window):
    n / p     : next / previous word
    SPACE     : start/stop recording frames for the current word
    r         : cycle to another reference video of the same word
    q         : quit and save

TIP: aim for ~500 samples per word (~20-25s of recording). Vary hand
position/angle slightly while holding the sign. For dynamic signs, hold the
most distinctive phase (usually the "mid" reference image).
"""
from __future__ import annotations

import csv
import os
from collections import Counter

import cv2
import numpy as np

from capture.landmarks import (
    make_hands, extract_landmarks_both_hands, draw_landmarks,
)
from training_words.words_list import INCAR_WORDS

DATA_PATH = os.path.join("data", "words_webcam.csv")
REF_DIR = os.path.join("data", "word_image_dataset_3frames")
TARGET_PER_WORD = 500
REF_HEIGHT = 200  # px height of the reference strip


def load_counts() -> Counter:
    counts: Counter = Counter()
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, newline="") as f:
            for row in csv.reader(f):
                if row:
                    counts[row[-1]] += 1
    return counts


def load_reference_sets(word: str) -> list[list[str]]:
    """Group the word's reference images by video (filename prefix before '_').

    Returns a list of [before, mid, after] path triples, one per source video.
    """
    d = os.path.join(REF_DIR, word)
    if not os.path.isdir(d):
        return []
    by_video: dict[str, dict[str, str]] = {}
    for fname in sorted(os.listdir(d)):
        base, ext = os.path.splitext(fname)
        if ext.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        if "_" not in base:
            continue
        vid, phase = base.rsplit("_", 1)
        by_video.setdefault(vid, {})[phase] = os.path.join(d, fname)
    sets = []
    for vid in sorted(by_video):
        phases = by_video[vid]
        triple = [phases.get(p) for p in ("before", "mid", "after")]
        triple = [p for p in triple if p]
        if triple:
            sets.append(triple)
    return sets


def build_reference_strip(paths: list[str], width: int) -> np.ndarray:
    """Load reference images, resize to REF_HEIGHT, and tile them horizontally."""
    tiles = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w * REF_HEIGHT / h), REF_HEIGHT))
        tiles.append(img)
    if not tiles:
        strip = np.zeros((REF_HEIGHT, width, 3), dtype=np.uint8)
        cv2.putText(strip, "no reference images", (10, REF_HEIGHT // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return strip
    strip = cv2.hconcat(tiles)
    # pad or crop to the webcam frame width so vconcat works
    if strip.shape[1] < width:
        pad = np.zeros((REF_HEIGHT, width - strip.shape[1], 3), dtype=np.uint8)
        strip = cv2.hconcat([strip, pad])
    else:
        strip = strip[:, :width]
    for i, label in enumerate(("before", "mid", "after")[: len(tiles)]):
        x = i * tiles[0].shape[1] + 5
        cv2.putText(strip, label, (x, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1)
    return strip


def main() -> None:
    os.makedirs("data", exist_ok=True)
    counts = load_counts()
    total_start = sum(counts.values())
    print(f"Loaded {total_start} existing samples across {len(counts)} words.")
    print(f"Vocabulary: {len(INCAR_WORDS)} words, target {TARGET_PER_WORD} samples each.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam (index 0).")

    hands = make_hands(max_num_hands=2)
    out = open(DATA_PATH, "a", newline="")
    writer = csv.writer(out)

    word_idx = 0
    ref_idx = 0
    recording = False
    ref_sets = load_reference_sets(INCAR_WORDS[word_idx])
    ref_strip = None  # built lazily once we know the frame width

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            word = INCAR_WORDS[word_idx]

            # Extract from the RAW (unflipped) frame — train == test invariant.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            feats = extract_landmarks_both_hands(results)

            draw_landmarks(frame, results)
            frame = cv2.flip(frame, 1)  # mirror preview only
            hand_seen = feats is not None

            if recording and hand_seen:
                writer.writerow([*feats.tolist(), word])
                counts[word] += 1

            # ── overlay ──────────────────────────────────────────
            n = counts.get(word, 0)
            done = "DONE " if n >= TARGET_PER_WORD else ""
            status = "REC" if recording else "paused"
            color = (0, 0, 255) if recording else (0, 180, 0)
            cv2.putText(frame,
                        f"[{word_idx + 1}/{len(INCAR_WORDS)}] {word.upper()}  "
                        f"{status}  samples: {n}/{TARGET_PER_WORD} {done}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, "n/p=next/prev word  SPACE=rec  r=other ref  q=quit",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            if not hand_seen:
                cv2.putText(frame, "no hand detected", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if ref_strip is None or ref_strip.shape[1] != frame.shape[1]:
                paths = ref_sets[ref_idx % len(ref_sets)] if ref_sets else []
                ref_strip = build_reference_strip(paths, frame.shape[1])
            display = cv2.vconcat([ref_strip, frame])

            cv2.imshow("collect words  (top: how to sign it)", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord(" "):
                recording = not recording
            elif key in (ord("n"), ord("p")):
                word_idx = (word_idx + (1 if key == ord("n") else -1)) % len(INCAR_WORDS)
                recording = False
                ref_idx = 0
                ref_sets = load_reference_sets(INCAR_WORDS[word_idx])
                ref_strip = None
            elif key == ord("r") and ref_sets:
                ref_idx += 1
                ref_strip = None
    finally:
        out.close()
        cap.release()
        hands.close()
        cv2.destroyAllWindows()

    final = load_counts()
    print("\n=== samples per word ===")
    for word in INCAR_WORDS:
        n = final.get(word, 0)
        flag = "" if n >= TARGET_PER_WORD else f"  <-- need {TARGET_PER_WORD - n} more"
        print(f"  {word.upper():<12} {n}{flag}")
    print(f"total: {sum(final.values())}  (added {sum(final.values()) - total_start})")
    print(f"saved -> {DATA_PATH}")
    print("\nNext: python -m training_words.train_words --data data/words_webcam.csv")


if __name__ == "__main__":
    main()
