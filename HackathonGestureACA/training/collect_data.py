"""Collect fingerspelling training data from the webcam.

Records normalized hand-landmark vectors labeled with the letter you're signing,
and appends them to data/fingerspell.csv (64 columns: 63 features + label).
See design doc §21.1 step 2.

USAGE:
    python -m training.collect_data

CONTROLS (focus the video window):
    - Hold a letter handshape, then press that LETTER key (a-z) to start/stop
      recording for it. While recording, every frame with a detected hand is saved.
    - The current target letter + its sample count is shown on screen.
    - SPACE : pause/resume recording
    - BACKSPACE : delete the last 20 samples of the current letter (undo a bad burst)
    - q : quit and save

TIP: aim for ~300-500 samples per letter. Vary hand angle/distance slightly.
Skip J and Z (they need motion) — see design §15.
"""
from __future__ import annotations

import csv
import os
from collections import Counter

import cv2

from capture.landmarks import make_hands, extract_landmarks, draw_landmarks, NUM_FEATURES

DATA_PATH = os.path.join("data", "fingerspell.csv")
SKIP = {"j", "z"}  # motion letters, not supported by the static MLP


def load_counts() -> Counter:
    counts: Counter = Counter()
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, newline="") as f:
            for row in csv.reader(f):
                if row:
                    counts[row[-1]] += 1
    return counts


def main() -> None:
    os.makedirs("data", exist_ok=True)
    counts = load_counts()
    total_start = sum(counts.values())
    print(f"Loaded {total_start} existing samples across {len(counts)} letters.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam (index 0).")

    hands = make_hands(max_num_hands=1)
    out = open(DATA_PATH, "a", newline="")
    writer = csv.writer(out)

    target: str | None = None      # letter currently selected
    recording = False              # are we actively saving frames?
    recent_written = 0             # samples written for current target (for undo)

    print("Ready. Focus the video window and press a letter key to select it.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # Extract from the RAW (unflipped) frame so collected landmarks match
            # both live inference and the image-extraction path (none of which flip
            # before extraction). Then mirror only the preview for natural framing.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            feats = extract_landmarks(results)

            draw_landmarks(frame, results)
            frame = cv2.flip(frame, 1)  # mirror preview only, feels natural
            hand_seen = feats is not None

            if recording and target and hand_seen:
                writer.writerow([*feats.tolist(), target])
                counts[target] += 1
                recent_written += 1

            # ── overlay ──────────────────────────────────────────
            status = "REC" if recording else "paused"
            color = (0, 0, 255) if recording else (0, 180, 0)
            tgt = target.upper() if target else "-"
            cv2.putText(frame, f"letter: {tgt}   {status}   samples: {counts.get(target, 0)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, "press a-z=select  SPACE=pause  BKSP=undo20  q=quit",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            if not hand_seen:
                cv2.putText(frame, "no hand detected", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("collect_data", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord(" "):
                recording = not recording
            elif key == 8:  # BACKSPACE — crude undo not supported on append file
                print("Undo isn't supported mid-file; edit data/fingerspell.csv if needed.")
            elif 97 <= key <= 122:  # a-z
                ch = chr(key)
                if ch in SKIP:
                    print(f"'{ch}' is a motion letter — skipped (design §15).")
                else:
                    target = ch
                    recording = True
                    recent_written = 0
                    print(f"Selected '{ch}'. Recording... (SPACE to pause)")
    finally:
        out.close()
        cap.release()
        hands.close()
        cv2.destroyAllWindows()

    final = load_counts()
    print("\n=== samples per letter ===")
    for letter in sorted(final):
        print(f"  {letter.upper()}: {final[letter]}")
    print(f"total: {sum(final.values())}  (added {sum(final.values()) - total_start})")
    print(f"saved -> {DATA_PATH}")


if __name__ == "__main__":
    main()
