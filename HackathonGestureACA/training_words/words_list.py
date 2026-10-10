"""The in-car demo vocabulary.

Chosen so the LLM composer can build real car queries from them
(e.g. TAKE ME RESTAURANT -> "Hey BMW, take me to a restaurant.").
Every word exists in data/word_image_dataset_3frames/ so reference
images of real signers are available during collection.

NOTE: "navigate" has no folder in the reference dataset (it is not a
standard single ASL sign) — sign TAKE + ME instead, which the composer
turns into "navigate me to ...".
"""

INCAR_WORDS = [
    # questions / intent
    "what", "when", "how", "find",
    # navigation
    "near", "take", "me",
    # destinations
    "restaurant", "hospital",
]

assert len(INCAR_WORDS) == 9
