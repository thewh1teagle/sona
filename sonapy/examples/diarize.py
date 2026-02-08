# /// script
# requires-python = ">=3.12"
# dependencies = ["sonapy"]
#
# [tool.uv.sources]
# sonapy = { path = "../" }
# ///
"""
Transcribe with speaker diarization.

Requires sona-diarize binary on PATH or next to sona.

Setup:
  wget https://github.com/thewh1teagle/sona/releases/download/v0.1.1/sona-diarize-darwin-arm64
  wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin
  wget https://huggingface.co/altunenes/parakeet-rs/resolve/main/diar_streaming_sortformer_4spk-v2.1.onnx
  wget https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/6_speakers.wav

Run:
  uv run examples/diarize.py ggml-tiny.bin diar_streaming_sortformer_4spk-v2.1.onnx 6_speakers.wav
"""

import sys
from sonapy import Sona


def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <whisper-model.bin> <diarize-model.onnx> <audio.wav>")
        sys.exit(1)

    whisper_model = sys.argv[1]
    diarize_model = sys.argv[2]
    audio_path = sys.argv[3]

    with Sona() as sona:
        print(f"Sona running on port {sona.port}")

        print(f"Loading whisper model: {whisper_model}")
        sona.load_model(whisper_model)

        # Transcribe with diarization — verbose_json includes speaker field
        print(f"Transcribing with diarization ({diarize_model})...")
        result = sona.transcribe(
            audio_path,
            response_format="verbose_json",
            diarize_model=diarize_model,
        )

        for seg in result["segments"]:
            speaker = f"Speaker {seg['speaker']}" if seg.get("speaker") is not None else "UNKNOWN"
            print(f"  [{seg['start']:.1f}s - {seg['end']:.1f}s] {speaker}: {seg['text']}")


if __name__ == "__main__":
    main()
