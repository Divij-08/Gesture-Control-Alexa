"""recognizer/ — landmark stream → token buffer. Owner: A.

Contract (see INTERFACES.md §2):
    emits TokenBuffer = list[str] when the signer signs DONE,
    in signing order, interleaved (CTC glosses + fingerspelled words). See design §7.

Holds:
    - fingerspell/  : MLP per-frame letter classifier (+ confusable-set trick)
    - ctc/          : BiLSTM+CTC continuous gloss recognizer
    - SPELL-mode state machine (design §7 diagram)
    - confidence threshold + N-stable-frames debounce gates

Depends on: trained weights in data/ (git-ignored).
"""
