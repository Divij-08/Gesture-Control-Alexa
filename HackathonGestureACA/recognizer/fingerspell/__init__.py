"""recognizer/fingerspell/ — per-frame MLP letter classifier. Owner: A.

One hand pose (63 landmark numbers) -> one letter (A-Z, skip J/Z; see design §21.1).
Includes the confusable-set trick (secondary classifiers for {M,N,S},{D,R,U},{T,K}).
Trained by training/train_fingerspell.py; weights saved here (git-ignored).
"""
