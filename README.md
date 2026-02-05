# Sona

Sona is a local transcription runner built on `whisper.cpp`.

It is designed to be spawned and owned by another process (desktop app, Python script, etc.), with an OpenAI-compatible HTTP API as the transport.

## What It Does

- Cross-platform binary (`macOS`, `Linux`, `Windows`)
- GPU-accelerated `whisper.cpp` backend
- OpenAI-compatible transcription endpoint: `POST /v1/audio/transcriptions`
- Runtime model management via API:
  - `POST /v1/models/load`
  - `DELETE /v1/models`
  - `GET /v1/models`
- Lifecycle endpoints:
  - `GET /health` (process alive)
  - `GET /ready` (model loaded or not)
- Dynamic port support: `sona serve --port 0` (OS assigns free port)
- Machine-readable ready signal on stdout:
  - `{"status":"ready","port":52341}`
- Response formats: `json`, `text`, `verbose_json`, `srt`, `vtt`
- Optional NDJSON streaming (`stream=true`) with progress + segment events

## Quick Start

1) Download a release binary:

https://github.com/thewh1teagle/sona/releases

2) Download a model:

```console
./sona pull https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin
```

3) Start Sona (no model required at startup):

```console
./sona serve --port 0
```

It prints one ready line to stdout with the actual bound port, for example:

```json
{"status":"ready","port":52341}
```

4) Load model through API:

```console
curl -X POST http://localhost:52341/v1/models/load \
  -H "content-type: application/json" \
  -d '{"path":"./ggml-base.bin"}'
```

5) Transcribe:

```console
curl -X POST http://localhost:52341/v1/audio/transcriptions \
  -F "file=@samples/jfk.wav" \
  -F "response_format=verbose_json"
```

## OpenAI Client Compatibility

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:52341/v1", api_key="sona")

with open("samples/jfk.wav", "rb") as f:
    result = client.audio.transcriptions.create(
        model="ignored-by-sona",
        file=f,
        response_format="text",
    )

print(result)
```

## Notes

- Sona handles one transcription at a time per process; concurrent requests return `429`.
- Max upload size is `1 GB`.
- Non-WAV/native audio is converted via `ffmpeg` automatically (system `ffmpeg` or bundled binary next to `sona`).
- API docs:
  - `GET /docs`
  - `GET /openapi.json`
