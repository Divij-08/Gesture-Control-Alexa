"""training/ — data collection + model training scripts. Owner: A.

Planned scripts (see design §21):
    collect_data.py       webcam -> landmarks -> labeled rows/clips in data/
    train_fingerspell.py  MLP letter classifier (+ confusable-set) -> §21.1
    train_ctc.py          BiLSTM+CTC continuous gloss recognizer   -> §21.2

Adapt the reference repo's TrainingDataCollection.py loop, but SAVE LANDMARKS,
not thresholded images.
"""
