# Sona Architecture

## Overview

Sona is a single-process Go binary with two modes:

- `sona transcribe <model.bin> <audio>`: one-shot local transcription.
- `sona serve [model.bin] --port <n>`: long-running HTTP runner.

The current server model is "runner, not shared service": one owner process spawns Sona, manages lifecycle, and sends transcription requests over HTTP.

## Runtime Components

- `cmd/sona/*`
  - CLI commands (`transcribe`, `serve`, `pull`).
- `internal/audio`
  - Converts input audio to `16kHz` mono `float32`.
  - Fast path: native PCM WAV (`internal/wav`).
  - Fallback: `ffmpeg` conversion.
- `internal/whisper`
  - CGo wrapper over `whisper.cpp`.
  - Exposes segments, progress callbacks, and abort callbacks.
- `internal/server`
  - HTTP routes, model lifecycle, concurrency control, and graceful shutdown.
- `sonapy/src/sonapy`
  - Python helper that spawns `sona serve --port 0`, waits for stdout ready signal, and calls the HTTP API.

## Server Lifecycle

1. `ListenAndServe` binds TCP (`--port 0` supported for auto-assigned port).
2. It prints one machine-readable line to stdout:
   - `{"status":"ready","port":<actual-port>}`
3. HTTP server starts and handles API traffic.
4. On `SIGINT`/`SIGTERM`:
   - stop accepting new connections (`http.Server.Shutdown`, 30s timeout)
   - unload model (`whisper.Context.Close`)
   - exit cleanly

## API Surface

- `GET /health`
  - Always `200`, process is alive.
- `GET /ready`
  - `200` when model is loaded.
  - `503` when no model is loaded.
- `POST /v1/models/load` with JSON body `{"path":"..."}`
  - Loads model from disk (replaces currently loaded model if any).
- `DELETE /v1/models`
  - Unloads current model; idempotent.
- `GET /v1/models`
  - Returns OpenAI-style model list with 0 or 1 loaded model.
- `POST /v1/audio/transcriptions`
  - Multipart file upload + form options:
    - `response_format`: `json` (default), `text`, `verbose_json`, `srt`, `vtt`
    - `stream`: `true|false` (NDJSON event stream when true)
    - `language`, `detect_language`, `prompt`, `enhance_audio`

Docs endpoints are served at `/docs` and `/openapi.json`.

## Transcription Execution Flow

1. `handleTranscription` takes a global mutex with `TryLock`.
2. If busy, returns `429` (no queue).
3. Validates model loaded; otherwise `503`.
4. Reads multipart `file` (max upload: `1 GB`).
5. Decodes audio (`internal/audio.ReadWithOptions`).
6. Calls `Context.TranscribeStream(...)`:
   - non-stream mode still uses stream-capable whisper path
   - client disconnect toggles abort callback for cancellation
7. Formats output by `response_format`:
   - `json`: `{ "text": "..." }`
   - `verbose_json`: full text + timestamped segments
   - `text`, `srt`, `vtt`: plain text subtitle/string responses

### Streaming Mode

When `stream=true`, `Content-Type: application/x-ndjson` is returned.

Event types emitted as JSON lines:

- `progress` (`progress: 0-100`)
- `segment` (`start`, `end`, `text`)
- `result` (`text`)
- `error` (`message`, if inference fails before disconnect)

Closing the client connection cancels inference through the whisper abort callback.

## Concurrency Model

- One mutex protects model state and inference path.
- Effective behavior:
  - one model loaded at a time
  - one transcription at a time per process
  - concurrent transcription requests return `429`
- Scale-out is process-level (run multiple Sona instances).

## Build and Packaging

- Whisper dependency is pinned by `.whisper.cpp-commit`.
- Platform static libs are downloaded to `third_party/lib` by `scripts/download-libs.py`.
- Headers are fetched to `third_party/include` by `scripts/fetch-headers.py`.
- Go binary links against platform-specific whisper/ggml libs.
- Release packaging (`scripts/package-release.py`) bundles `sona` plus `ffmpeg`.

## Current Boundaries

- No auth/multi-tenant concerns (runner model assumes single owner).
- No internal job queue or async job IDs (busy returns `429`).
- No daemon/service-manager integration (systemd/launchd/windows service).
- No in-process bindings for non-Go runtimes; integrations use HTTP (optionally via `sonapy`).
