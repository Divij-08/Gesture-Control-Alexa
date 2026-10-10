"""Diagnostic: is MediaPipe Hands detecting your hand at all?

Standalone probe with NO model and NO thresholds beyond detection. It prints, per
second, how many of the last frames had a hand, and overlays a big HAND / NO HAND
banner + the raw detection state. Use it to isolate "MediaPipe can't see my hand"
from model/threshold problems.

It runs detection at a LOW min_detection_confidence (0.3) so we can tell whether
the default 0.6 in capture/landmarks.py is simply too strict for your lighting.

USAGE:
    python -m capture.diagnose_hand
    python -m capture.diagnose_hand --conf 0.5   # try a specific threshold
    python -m capture.diagnose_hand --cam 1      # try a different camera index

Watch for:
  * "NO HAND" even with your hand centered + well-lit  -> lighting/framing (move
    closer, palm toward camera, brighter, plain background).
  * Detects at --conf 0.3 but not 0.6              -> lower make_hands min_conf.
  * Detects fine here but not in live_test         -> the issue is downstream, not detection.
"""
from __future__ import annotations

import argparse
import collections

import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.3, help="min_detection_confidence")
    ap.add_argument("--cam", type=int, default=0, help="camera index")
    ap.add_argument("--max-hands", type=int, default=2)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.cam}. Try --cam 1.")

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=args.max_hands,
        min_detection_confidence=args.conf,
        min_tracking_confidence=args.conf,
    )

    window = collections.deque(maxlen=60)   # last ~60 frames
    print(f"Probing at min_detection_confidence={args.conf}, camera {args.cam}. q to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            n = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
            window.append(1 if n else 0)

            if results.multi_hand_landmarks:
                for hl in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hl, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style())

            frame = cv2.flip(frame, 1)
            rate = 100 * sum(window) / len(window)
            detected = n > 0
            banner = f"HAND x{n}" if detected else "NO HAND"
            color = (0, 200, 0) if detected else (0, 0, 255)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (0, 0, 0), -1)
            cv2.putText(frame, banner, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
            cv2.putText(frame, f"detection rate (last 60f): {rate:3.0f}%   conf={args.conf}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("hand detection diagnostic (q=quit)", frame)
            if (cv2.waitKey(5) & 0xFF) == ord("q"):
                break
    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
