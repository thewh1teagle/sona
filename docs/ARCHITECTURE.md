# Sona Architecture

## Overview

Sona is a single-process Go application with two modes:

- `sona transcribe <model.bin> <audio>`: one-shot local transcription.
- `sona serve <model.bin>`: OpenAI-compatible HTTP server.

It is not a background daemon by default. The process lifetime is tied to the CLI invocation.

## Runtime components

- `cmd/sona/*`: CLI entrypoint and command wiring (Cobra).
- `internal/audio`: input decoding/conversion to 16kHz mono float32 samples.
  - Fast path: native PCM WAV via `internal/wav`.
  - Fallback: `ffmpeg` conversion for non-native formats.
- `internal/whisper`: cgo wrapper around whisper.cpp.
  - Platform link config in `whisper_linux.go`, `whisper_darwin.go`, `whisper_windows.go`.
  - Core wrapper in `whisper_cgo.go` (`New`, `Transcribe`, `Close`).
- `internal/server`: HTTP API surface.
  - `POST /v1/audio/transcriptions`
  - `GET /v1/models`

## Execution flow

### CLI transcription flow

1. Parse flags/args in `cmd/sona/commands.go`.
2. Decode audio (`internal/audio.ReadFile`).
3. Load model into a whisper context (`whisper.New`).
4. Run inference (`Context.Transcribe`).
5. Print text and exit.

### Server flow

1. Start process with `sona serve <model.bin>`.
2. Load one whisper context and keep it in memory.
3. For each request:
   - Read multipart `file` field (max 25 MB).
   - Decode/convert audio.
   - Transcribe with shared context.
   - Return JSON `{ "text": "..." }`.

## Concurrency model

- `internal/server.Server` protects the shared whisper context with a mutex.
- Effect: one transcription runs at a time per process instance.
- Scale-out today is process-level (run multiple instances), not intra-process parallel request execution.

## Build and packaging architecture

- Whisper dependency is pinned by `.whisper.cpp-commit`.
- Prebuilt static libs are produced per platform and published as GitHub release assets (`libraries-<commit7>`).
- `scripts/download-libs.py` fetches those libs into `third_party/lib`.
- `scripts/fetch-headers.py` fetches C headers into `third_party/include`.
- Go binary links statically against the downloaded whisper/ggml libs (platform-specific cgo flags).
- Release packaging (`scripts/package-release.py`) bundles the `sona` binary plus `ffmpeg`.

## Current boundaries

- No built-in daemon/service management (systemd/launchd/windows service not included).
- No installer package flow in repo yet (Homebrew/apt/choco/etc.).
- No self-update mechanism in the binary.
- API returns completed transcription only; no token/segment streaming endpoint yet.
- Python support is API-based today (OpenAI client works against `sona serve`), not an in-process Python module.

## Opportunities

- URL ingestion via `yt-dlp`: accept media URLs in CLI/API, fetch audio with `yt-dlp`, then route into the existing `ffmpeg` conversion + transcription pipeline.
