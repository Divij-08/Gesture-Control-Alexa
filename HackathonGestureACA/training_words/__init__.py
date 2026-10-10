"""Word-level sign-language training pipeline.

Extracts MediaPipe hand landmarks from image datasets (one folder per word),
trains a scikit-learn MLP to classify static word-level signs, and produces
a model for live webcam inference.

Mirrors the fingerspelling pipeline (training/) at word granularity.
"""
