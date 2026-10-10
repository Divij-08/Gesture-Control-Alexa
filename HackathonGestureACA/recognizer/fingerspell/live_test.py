"""Live real-time test of the trained fingerspelling MLP.

Loads recognizer/fingerspell/model.joblib and predicts the letter from the webcam
in real time, applying the two demo-safety gates from the design (§7, §16):
  - confidence threshold : ignore unsure predictions
  - N-stable-frames debounce : only "commit" a letter after it's been the top
    prediction for N consecutive frames (kills flicker/misfires)

It also assembles committed letters into a word buffer so you can see the
fingerspelling -> text idea working end to end.

USAGE:
    python -m recognizer.fingerspell.live_test

CONTROLS:
    SPACE : clear the word buffer
    q     : quit
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import joblib

from capture.landmarks import make_hands, extract_landmarks, draw_landmarks

MODEL_PATH = os.path.join("recognizer", "fingerspell", "model.joblib")
CONF_THRESHOLD = 0.80   # ignore predictions below this probability
STABLE_FRAMES = 15     # ~0.5s at 30fps: frames the same letter must persist to commit


def main() -> None:
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"No model at {MODEL_PATH}. Run: python -m training.train_fingerspell")
    bundle = joblib.load(MODEL_PATH)
    clf, labels = bundle["model"], bundle["labels"]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam (index 0).")
    hands = make_hands(max_num_hands=1)

    word = ""
    last_letter = None
    stable_count = 0
    committed_recently = None  # prevents re-committing the same held pose repeatedly

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # Extract landmarks from the RAW (unflipped) frame so the feature
            # orientation matches the training data (extract_from_images does NOT
            # flip). Flipping before extraction mirrors the hand and destroys
            # accuracy — see capture/landmarks.py "train == test" invariant.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            feats = extract_landmarks(results)

            # Draw the skeleton on the unflipped frame, THEN mirror for a natural
            # selfie-view preview — the skeleton flips with the image and stays aligned.
            draw_landmarks(frame, results)
            frame = cv2.flip(frame, 1)
            pred_letter, conf = None, 0.0
            if feats is not None:
                proba = clf.predict_proba([feats])[0]
                idx = int(np.argmax(proba))
                pred_letter, conf = labels[idx], float(proba[idx])

            # ── debounce + confidence gate ───────────────────────
            if pred_letter is not None and conf >= CONF_THRESHOLD:
                if pred_letter == last_letter:
                    stable_count += 1
                else:
                    last_letter = pred_letter
                    stable_count = 1
                    committed_recently = None

                if stable_count == STABLE_FRAMES and pred_letter != committed_recently:
                    word += pred_letter.upper()
                    committed_recently = pred_letter  # require a change before next commit
            else:
                last_letter = None
                stable_count = 0

            # ── overlay ──────────────────────────────────────────
            shown = pred_letter.upper() if pred_letter else "-"
            bar = "#" * min(stable_count, STABLE_FRAMES)
            cv2.putText(frame, f"pred: {shown}  conf: {conf:.2f}  [{bar}]",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
            cv2.putText(frame, f"WORD: {word}",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2)
            cv2.putText(frame, "SPACE=clear  q=quit",
                        (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("fingerspell live", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                word = ""
    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()

    print(f"final word buffer: {word}")


if __name__ == "__main__":
    main()
