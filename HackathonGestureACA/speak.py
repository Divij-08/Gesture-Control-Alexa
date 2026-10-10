#!/usr/bin/env python3
"""Text -> speech -> WAV -> speaker, in one file.

Distilled from AgenticAiDeviceFarmHostAgent/configuration/iot_agent/simple_agent.py
(the `text_to_speech` Polly->WAV writer and the `speaker` playback fallback chain).
It does exactly three things:

  1. synthesize the text with Amazon Polly (OutputFormat='pcm', 16 kHz mono),
  2. wrap the raw PCM in a WAV header and write a .wav file,
  3. play that .wav on the host speaker.

CREDENTIALS / ACCOUNT: nothing is hardcoded (same contract as composer/bedrock.py).
We build a plain boto3.Session(), so it uses AWS_PROFILE if you export one, else
the [default] profile. Region resolves from --region, then AWS_REGION /
AWS_DEFAULT_REGION, then the profile's config. Refresh your profile's creds (e.g.
via `ada`) before running.

Usage:
    python speak.py "Hey BMW, navigate home"
    python speak.py "Turn up the volume" --voice Joanna --keep out.wav
    echo "piped text" | python speak.py
"""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import tempfile
import uuid


# Polly PCM output is always signed 16-bit little-endian mono. 16 kHz matches the
# reference and keeps files small; Polly also supports 8000 / 22050 for 'pcm'.
_SAMPLE_RATE = 16000
_BITS_PER_SAMPLE = 16
_CHANNELS = 1


def _make_polly_client(region: str | None = None):
    """Build a Polly client from the ambient AWS environment.

    boto3.Session() with no profile_name uses AWS_PROFILE if set, otherwise the
    [default] profile, and runs that profile's credential_process if it has one.
    """
    import boto3  # imported lazily so --help etc. work without boto3 installed

    session = boto3.Session()
    region = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    return session.client("polly", region_name=region)


def _write_wav(pcm_data: bytes, output_file: str) -> None:
    """Wrap raw PCM in a 44-byte WAV header and write it out.

    Copied structurally from the reference: write placeholder RIFF/data sizes,
    stream the PCM, then seek back and patch in the real sizes.
    """
    byte_rate = _SAMPLE_RATE * _CHANNELS * (_BITS_PER_SAMPLE // 8)
    block_align = _CHANNELS * (_BITS_PER_SAMPLE // 8)

    with open(output_file, "wb") as wav_file:
        wav_file.write(b"RIFF")
        wav_file.write(b"\x00\x00\x00\x00")                 # ChunkSize (patched below)
        wav_file.write(b"WAVE")
        wav_file.write(b"fmt ")
        wav_file.write((16).to_bytes(4, "little"))          # Subchunk1Size (PCM)
        wav_file.write((1).to_bytes(2, "little"))           # AudioFormat = PCM
        wav_file.write((_CHANNELS).to_bytes(2, "little"))
        wav_file.write((_SAMPLE_RATE).to_bytes(4, "little"))
        wav_file.write((byte_rate).to_bytes(4, "little"))
        wav_file.write((block_align).to_bytes(2, "little"))
        wav_file.write((_BITS_PER_SAMPLE).to_bytes(2, "little"))
        wav_file.write(b"data")
        wav_file.write(b"\x00\x00\x00\x00")                 # Subchunk2Size (patched below)

        wav_file.write(pcm_data)

        file_size = wav_file.tell()
        wav_file.seek(4)
        wav_file.write((file_size - 8).to_bytes(4, "little"))   # RIFF chunk size
        wav_file.seek(40)
        wav_file.write((len(pcm_data)).to_bytes(4, "little"))   # data chunk size


def text_to_wav(
    text: str,
    output_file: str,
    voice_id: str = "Joanna",
    voice_language: str = "en-US",
    voice_engine: str = "neural",
    region: str | None = None,
) -> str:
    """Synthesize `text` with Polly and write a playable .wav. Returns output_file."""
    polly = _make_polly_client(region)
    response = polly.synthesize_speech(
        Text=text,
        TextType="text",
        OutputFormat="pcm",
        VoiceId=voice_id,
        SampleRate=str(_SAMPLE_RATE),
        LanguageCode=voice_language,
        Engine=voice_engine,
    )
    if "AudioStream" not in response:
        raise RuntimeError("Polly returned no AudioStream")

    with response["AudioStream"] as stream:
        pcm_data = stream.read()
    _write_wav(pcm_data, output_file)
    return output_file


def _command_exists(cmd: str) -> bool:
    from shutil import which

    return which(cmd) is not None


def play_wav(wav_file: str) -> None:
    """Play a .wav on the host speaker.

    macOS uses afplay. Linux tries pw-play (PipeWire) -> paplay (PulseAudio) ->
    aplay (ALSA) -> ffplay (ffmpeg), in that order, matching the reference.
    """
    system = platform.system()

    if system == "Darwin":  # macOS
        subprocess.run(["afplay", wav_file], check=True, timeout=60)
        return

    if system == "Linux":
        players = [
            ["pw-play", wav_file],
            ["paplay", wav_file],
            ["aplay", wav_file],
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", wav_file],
        ]
        for cmd in players:
            if not _command_exists(cmd[0]):
                continue
            try:
                subprocess.run(cmd, check=True, timeout=60)
                return
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                print(f"warning: {cmd[0]} failed: {e}", file=sys.stderr)
        raise RuntimeError(
            "No working audio player found (tried pw-play/paplay/aplay/ffplay)"
        )

    raise RuntimeError(f"Speaker playback not supported on {system}")


def speak(
    text: str,
    voice_id: str = "Joanna",
    voice_language: str = "en-US",
    voice_engine: str = "neural",
    region: str | None = None,
    keep_path: str | None = None,
) -> None:
    """End-to-end: text -> Polly -> WAV -> speaker.

    If keep_path is given the .wav is written there and left on disk; otherwise a
    temp file is used and removed afterward.
    """
    if keep_path:
        wav_file = keep_path
        cleanup = False
    else:
        wav_file = os.path.join(tempfile.gettempdir(), f"speak_{uuid.uuid4().hex}.wav")
        cleanup = True

    try:
        text_to_wav(text, wav_file, voice_id, voice_language, voice_engine, region)
        play_wav(wav_file)
    finally:
        if cleanup and os.path.exists(wav_file):
            os.remove(wav_file)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Speak text through the speaker via Amazon Polly."
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to speak. If omitted, read from stdin.",
    )
    parser.add_argument("--voice", default="Joanna", help="Polly VoiceId (default: Joanna)")
    parser.add_argument("--language", default="en-US", help="LanguageCode (default: en-US)")
    parser.add_argument(
        "--engine",
        default="neural",
        choices=["neural", "standard", "long-form", "generative"],
        help="Polly engine (default: neural)",
    )
    parser.add_argument("--region", default=None, help="AWS region (default: env / profile)")
    parser.add_argument(
        "--keep",
        metavar="PATH",
        default=None,
        help="Write the .wav to PATH and keep it (default: temp file, deleted after)",
    )
    args = parser.parse_args(argv)

    text = args.text if args.text is not None else sys.stdin.read()
    text = text.strip()
    if not text:
        parser.error("no text provided (pass an argument or pipe via stdin)")

    speak(
        text,
        voice_id=args.voice,
        voice_language=args.language,
        voice_engine=args.engine,
        region=args.region,
        keep_path=args.keep,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
