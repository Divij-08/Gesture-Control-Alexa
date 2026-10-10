"""aca_client/ — invoke_aca_live(): inject utterance into real ACA. Owner: B.

Contract (see INTERFACES.md §4):
    invoke_aca_live(text: str, thing_id: str, timeout_s: int = 30) -> AcaResponse

Reuses the device agent's EXISTING MQTT commands (design §9):
    1. subscribe to device/{thing_id}/response FIRST (unique client_id!)
    2. publish start_voice_session -> get session_id
    3. publish speak_in_session {text, session_id}
    4. read alexa_response off the matching correlation_id

Fallback ladder (design §16): live -> stub -> cached; report which rung in AcaResponse.mode.

SECURITY: read device certs from /etc/iot-agent/ at runtime. NEVER commit them.
Depends on: awsiotsdk.
"""
