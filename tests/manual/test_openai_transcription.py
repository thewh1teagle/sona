# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "openai==1.109.1",
# ]
# ///

"""
Manual OpenAI-client compatibility test for Sona transcription API.

Run in two terminals:

1) Terminal A: download a tiny Whisper model and start Sona server
   wget -O ggml-tiny.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin
   ./sona serve ./ggml-tiny.bin

2) Terminal B: download a sample audio and run this script
   wget -O samples/jfk.wav https://github.com/ggml-org/whisper.cpp/raw/master/samples/jfk.wav
   uv run tests/manual/test_openai_transcription.py

Optional:
   uv run tests/manual/test_openai_transcription.py --base-url http://localhost:11531/v1 --audio /path/to/audio.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path

from openai import OpenAI


DEFAULT_BASE_URL = "http://localhost:11531/v1"
DEFAULT_MODEL = "ggml-tiny.bin"


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual OpenAI transcription test against Sona")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default="sona", help="Dummy API key for OpenAI client")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model id sent to the API")
    parser.add_argument("--audio", default="samples/jfk.wav", help="Path to audio file")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(
            f"audio file not found: {audio_path}. Download one with wget first or pass --audio."
        )

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    with audio_path.open("rb") as f:
        result = client.audio.transcriptions.create(model=args.model, file=f)

    print("transcription response:")
    print(result)


if __name__ == "__main__":
    main()
