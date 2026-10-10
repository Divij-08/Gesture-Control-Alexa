"""Live real-time test of the trained word-level sign MLP.

Loads recognizer/words/model.joblib and predicts the word from the webcam
in real time, applying confidence threshold + stable-frames debounce.

Committed words are assembled into a sentence buffer on screen.

USAGE:
    python -m recognizer.words.live_test

CONTROLS:
    SPACE : clear the sentence buffer
    q     : quit
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import joblib

from capture.landmarks import make_hands, extract_landmarks_both_hands, draw_landmarks

MODEL_PATH = os.path.join("recognizer", "words", "model.joblib")
CONF_THRESHOLD = 0.75
STABLE_FRAMES = 12


def main() -> None:
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"No model at {MODEL_PATH}. Run: python -m training_words.train_words")
    bundle = joblib.load(MODEL_PATH)
    clf, labels = bundle["model"], bundle["labels"]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam (index 0).")
    hands = make_hands(max_num_hands=2)

    sentence = []
    last_word = None
    stable_count = 0
    committed_recently = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            feats = extract_landmarks_both_hands(results)

            draw_landmarks(frame, results)
            frame = cv2.flip(frame, 1)

            pred_word, conf = None, 0.0
            if feats is not None:
                proba = clf.predict_proba([feats])[0]
                idx = int(np.argmax(proba))
                pred_word, conf = labels[idx], float(proba[idx])

            if pred_word is not None and conf >= CONF_THRESHOLD:
                if pred_word == last_word:
                    stable_count += 1
                else:
                    last_word = pred_word
                    stable_count = 1
                    committed_recently = None

                if stable_count == STABLE_FRAMES and pred_word != committed_recently:
                    sentence.append(pred_word.upper())
                    committed_recently = pred_word
            else:
                last_word = None
                stable_count = 0

            shown = pred_word.upper() if pred_word else "-"
            bar = "#" * min(stable_count, STABLE_FRAMES)
            sentence_str = " ".join(sentence) if sentence else "(empty)"

            cv2.putText(frame, f"pred: {shown}  conf: {conf:.2f}  [{bar}]",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
            cv2.putText(frame, f"SENTENCE: {sentence_str}",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
            cv2.putText(frame, "SPACE=clear  q=quit",
                        (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("word sign live", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                sentence = []
    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()

    print(f"final sentence: {' '.join(sentence)}")


if __name__ == "__main__":
    main()
