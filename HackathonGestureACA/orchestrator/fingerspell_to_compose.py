"""Integration: fingerspell recognizer -> composer (A -> B seam end to end).

Demonstrates the §11-B headline flow — a command gloss plus a fingerspelled
proper noun — turned into the final wake-word-prefixed utterance the ACA leg
would inject:

    command "NAV" + fingerspell C-U-B-B-O-N-P-A-R-K
        -> tokens ["NAV", "CUBBONPARK"]
        -> compose(tokens)
        -> "Hey BMW, navigate to Cubbon Park."   (live LLM)

Why a command gloss is supplied here instead of recognized: the CTC lexical path
(WHERE/NAV/CALL...) is not built yet (design P3). Fingerspelling only produces the
proper-noun slot, so we stand in the command gloss via --command / a keypress.
When CTC lands, it replaces this stand-in and appends to the SAME token buffer.

MODES
  --sim WORD        Skip the camera: use WORD as the fingerspelled result.
                    Runs the full token-buffer -> compose -> string path.
                    (No webcam, no model file needed. Good for CI / no-camera.)
  (default)         Live webcam: fingerspell a word with the trained MLP, press
                    ENTER to finish the word, then it composes and prints.

USAGE
  # simulated (works anywhere):
  python -m orchestrator.fingerspell_to_compose --command NAV --sim CUBBONPARK
  GESTUREACA_COMPOSER_OFFLINE=1 python -m orchestrator.fingerspell_to_compose --command WHERE --sim TAJMAHAL

  # live webcam (needs recognizer/fingerspell/model.joblib + a camera):
  python -m orchestrator.fingerspell_to_compose --command NAV

The composer path (live Bedrock LLM vs rule-based) is chosen by the composer's
own env flags (GESTUREACA_COMPOSER_OFFLINE, AWS_PROFILE); this module doesn't
decide it — it just calls compose().
"""
from __future__ import annotations

import argparse
import os

from composer.compose import compose


def build_tokens(command: str | None, spelled_word: str) -> list[str]:
    """Assemble the TokenBuffer in signing order: command gloss then spelled slot.

    The spelled word is an UPPERCASE proper-noun token (the composer treats a
    multi-word or all-caps token as a fingerspelled name — see composer/rules.py
    and the §8 prompt). The command gloss (if any) precedes it, mirroring how the
    CTC path will interleave in signing order.
    """
    tokens: list[str] = []
    if command:
        tokens.append(command.strip().upper())
    if spelled_word:
        tokens.append(spelled_word.strip().upper())
    return tokens


def run_once(command: str | None, spelled_word: str, wake_word: str,
             enabled_speak: bool = True) -> str:
    """tokens -> compose -> final string, with the intermediate shapes printed."""
    tokens = build_tokens(command, spelled_word)
    print(f"\n  spelled word : {spelled_word!r}")
    print(f"  command gloss: {command!r}  (stand-in; CTC not built yet)")
    print(f"  TokenBuffer  : {tokens}")
    utterance = compose(tokens, wake_word)
    print(f"\n  >>> UTTERANCE: {utterance}\n")

    # Speak the composer output aloud (never crashes the run on TTS failure).
    if enabled_speak and utterance:
        try:
            from speak import speak
            speak(utterance)
        except Exception as e:  # noqa: BLE001 — TTS must not break the run
            print(f"[voice] speak failed ({type(e).__name__}: {e}); continuing.")
    return utterance


# ── live webcam capture (reuses the trained fingerspelling MLP) ───────

def capture_word_from_webcam() -> str:
    """Fingerspell a word at the webcam and return it. ENTER = finish, q = abort.

    Reuses capture/landmarks.py + recognizer/fingerspell/model.joblib with the
    same confidence + N-stable-frames gates as live_test.py. Extraction is on the
    UNFLIPPED frame (train == test); the preview is mirrored.
    """
    import cv2
    import numpy as np
    import joblib
    from capture.landmarks import make_hands, extract_landmarks, draw_landmarks

    model_path = os.path.join("recognizer", "fingerspell", "model.joblib")
    if not os.path.exists(model_path):
        raise SystemExit(f"No model at {model_path}. Run: python -m training.train_fingerspell")
    bundle = joblib.load(model_path)
    clf, labels = bundle["model"], bundle["labels"]

    CONF_THRESHOLD, STABLE_FRAMES = 0.80, 15
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam (index 0).")
    hands = make_hands(max_num_hands=1)

    word = ""
    last_letter, stable_count, committed_recently = None, 0, None
    print("Fingerspell the proper noun. ENTER = done, SPACE = clear, q = abort.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            feats = extract_landmarks(results)
            draw_landmarks(frame, results)
            frame = cv2.flip(frame, 1)

            pred_letter, conf = None, 0.0
            if feats is not None:
                proba = clf.predict_proba([feats])[0]
                idx = int(np.argmax(proba))
                pred_letter, conf = labels[idx], float(proba[idx])

            if pred_letter is not None and conf >= CONF_THRESHOLD:
                if pred_letter == last_letter:
                    stable_count += 1
                else:
                    last_letter, stable_count, committed_recently = pred_letter, 1, None
                if stable_count == STABLE_FRAMES and pred_letter != committed_recently:
                    word += pred_letter.upper()
                    committed_recently = pred_letter
            else:
                last_letter, stable_count = None, 0

            shown = pred_letter.upper() if pred_letter else "-"
            cv2.putText(frame, f"pred: {shown}  conf: {conf:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
            cv2.putText(frame, f"WORD: {word}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2)
            cv2.putText(frame, "ENTER=done  SPACE=clear  q=abort", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow("fingerspell -> compose", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                word = ""
                break
            elif key == ord(" "):
                word = ""
            elif key in (13, 10):  # ENTER / RETURN
                break
    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()
    return word


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--command", default="NAV",
                    help="stand-in command gloss (e.g. NAV, WHERE, CALL); '' for none")
    ap.add_argument("--sim", metavar="WORD", default=None,
                    help="skip the camera; use WORD as the fingerspelled result")
    ap.add_argument("--wake-word", default=os.environ.get("GESTUREACA_WAKE_WORD", "Hey BMW"))
    ap.add_argument("--no-speak", action="store_true",
                    help="don't speak the utterance aloud (Polly TTS is on by default)")
    args = ap.parse_args()

    spelled = args.sim if args.sim is not None else capture_word_from_webcam()
    if not spelled:
        print("No word captured — nothing to compose.")
        raise SystemExit(1)

    command = args.command or None
    run_once(command, spelled, args.wake_word, enabled_speak=not args.no_speak)


if __name__ == "__main__":
    main()
