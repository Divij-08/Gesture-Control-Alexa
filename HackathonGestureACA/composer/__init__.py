"""composer/ — token buffer -> English utterance via Bedrock/Claude. Owner: B.

Contract (see INTERFACES.md §3):
    compose(tokens: list[str], wake_word: str = "Hey BMW") -> str

The LLM does three jobs in one call (design §8):
    1. reorder ASL word order -> English
    2. insert dropped function words (is/the/a/to/in)
    3. prepend the wake word
It is also the error-repair safety net for imperfect recognition.

Develop against a hand-typed token list; no camera needed.
Depends on: boto3 (Bedrock).
"""
