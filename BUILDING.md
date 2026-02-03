# Building Sonara

## Architecture

The C library (whisper.cpp) and the Go binary are built separately:

1. **`.whisper.cpp-commit`** is the single source of truth for the whisper.cpp version. All scripts read from it.
2. **`scripts/build-libs.py`** clones whisper.cpp at that commit, builds static `.a` files, and uploads them to a GitHub release tagged `libraries-{commit[:7]}`.
3. **`scripts/download-libs.py`** downloads the prebuilt `.a` files for the current platform from that release into `third_party/lib/`.
4. **`scripts/fetch-headers.py`** fetches the C headers into `third_party/include/` (these are checked into git).
5. **`go build`** links the Go code against `third_party/include/` and `third_party/lib/`.

This separation means contributors never need to build whisper.cpp locally -- they just run the download script.

## Prerequisites

- [Go](https://go.dev/dl/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (runs Python build scripts)

## Quick start

```bash
uv run scripts/fetch-headers.py
uv run scripts/download-libs.py
go build -o sonara ./cmd/sonara/
```

## Bumping whisper.cpp

1. Update the commit hash in `.whisper.cpp-commit`
2. Run `uv run scripts/fetch-headers.py` and commit the updated headers
3. Trigger the `Build whisper.cpp libs` workflow (or run `uv run scripts/build-libs.py --upload` locally)
