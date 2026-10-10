"""ui/ — FastAPI + WebSocket display surface. Owner: C.

A display only (design §6): shows camera feed + landmark overlay + live tokens +
current mode + composed sentence + ACA reply. Consumes UiState over a WebSocket
from the orchestrator. No logic lives here.

Depends on: fastapi, uvicorn, websockets.
"""
