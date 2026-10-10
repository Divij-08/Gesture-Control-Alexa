"""orchestrator/ — the state machine that wires everything. Owner: C.

Drives: capture -> recognizer -> composer -> aca_client, streaming UiState to the UI.
States (design §7): IDLE -> SIGNING -> (SPELLING) -> COMPOSE -> CONFIRM -> DISPATCH -> SHOW.
Owns the confidence/debounce gates coordination and the ACA fallback ladder.

Develop against mocks: a fake recognizer emitting a canned TokenBuffer, and a fake
invoke_aca_live returning a stub AcaResponse. Integration lands here.
"""
