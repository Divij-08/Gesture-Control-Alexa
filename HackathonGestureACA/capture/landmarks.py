"""Shared MediaPipe hand-landmark extraction + normalization.

Both data collection (training/collect_data.py) and live inference
(recognizer/fingerspell/live_test.py) import from here, so the feature
vector is guaranteed identical in train and test. See design doc §21.0.

Feature vector: 21 hand landmarks x (x, y, z) = 63 numbers, normalized:
  - translate so the wrist (landmark 0) is the origin
  - scale by the hand's bounding-box span
This makes the shape invariant to where the hand is in the frame and how big it is.
"""
from __future__ import annotations

import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

NUM_FEATURES = 63  # 21 landmarks * 3 coords (single hand — fingerspelling)
NUM_FEATURES_BOTH = 126  # 2 hands * 21 landmarks * 3 coords (word-level signs)


def make_hands(max_num_hands: int = 1, min_conf: float = 0.6,
               static_image_mode: bool = False):
    """Create a MediaPipe Hands detector. One hand for fingerspelling.

    static_image_mode=False (default) tracks across frames — use for webcam.
    static_image_mode=True treats each input as an independent still — use when
    extracting landmarks from a folder of unrelated images (see
    training/extract_from_images.py).
    """
    return mp_hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_conf,
        min_tracking_confidence=min_conf,
    )


def extract_landmarks(results) -> np.ndarray | None:
    """From a MediaPipe result, return a normalized 63-vector, or None if no hand.

    Normalization: wrist-relative + scale-normalized (see module docstring).
    """
    if not results.multi_hand_landmarks:
        return None

    hand = results.multi_hand_landmarks[0]
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float32)  # (21, 3)

    # 1) translate: wrist (landmark 0) -> origin
    pts = pts - pts[0]

    # 2) scale: divide by the max span so hand size doesn't matter
    span = np.linalg.norm(pts, axis=1).max()
    if span > 1e-6:
        pts = pts / span

    return pts.flatten()  # (63,)


def mirror_landmarks(feats: np.ndarray) -> np.ndarray:
    """Return the left/right-hand mirror of a 63-vector by negating each x coord.

    A horizontal image flip maps (x, y, z) -> (-x, y, z) for every landmark. Our
    features are already wrist-relative (wrist at origin) and scaled by a span that
    is invariant under x-negation, so negating x is the EXACT geometric mirror --
    identical to flipping the image and re-running MediaPipe, but instant and with
    no new detection misses. Used for handedness-invariant training augmentation:
    the mirror of a valid handshape is the same letter signed with the other hand.
    """
    m = feats.copy()
    m[0::3] = -m[0::3]  # x is at flattened indices 0, 3, 6, ... (21 landmarks)
    return m


def extract_landmarks_both_hands(results) -> np.ndarray | None:
    """From a MediaPipe result, return a normalized 126-vector for two hands, or None.

    Returns a concatenation of [left_hand_63, right_hand_63]. If only one hand is
    detected, the missing hand's slot is zero-filled. Hand assignment uses
    MediaPipe's handedness classification (label "Left"/"Right" — note these are
    from the camera's perspective, i.e. mirrored relative to the signer).

    For word-level signs that use both hands (recognizer/words/). Each hand is
    normalized independently the same way extract_landmarks() does a single hand,
    so the train==test invariant holds for the word path too.
    """
    if not results.multi_hand_landmarks:
        return None

    left = np.zeros(NUM_FEATURES, dtype=np.float32)
    right = np.zeros(NUM_FEATURES, dtype=np.float32)

    for hand_landmarks, handedness_info in zip(
        results.multi_hand_landmarks, results.multi_handedness
    ):
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
        pts = pts - pts[0]
        span = np.linalg.norm(pts, axis=1).max()
        if span > 1e-6:
            pts = pts / span
        flat = pts.flatten()

        label = handedness_info.classification[0].label
        if label == "Left":
            left = flat
        else:
            right = flat

    return np.concatenate([left, right])


def mirror_landmarks_both(feats: np.ndarray) -> np.ndarray:
    """Mirror a 126-vector: negate x coords AND swap the left/right hand slots.

    A horizontal flip mirrors each hand (negate x) and also swaps which hand is
    on which side — the two-hand analog of mirror_landmarks(). Used for
    handedness-invariant word-model augmentation.
    """
    left = feats[:NUM_FEATURES].copy()
    right = feats[NUM_FEATURES:].copy()
    left[0::3] = -left[0::3]
    right[0::3] = -right[0::3]
    return np.concatenate([right, left])


def draw_landmarks(image, results) -> None:
    """Draw the hand skeleton onto a BGR image in-place (for the preview window)."""
    if not results.multi_hand_landmarks:
        return
    for hand in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand,
            mp_hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style(),
        )
