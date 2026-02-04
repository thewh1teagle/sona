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

On Windows, build with cgo enabled and MinGW available (MSYS2 `MINGW64` shell is the easiest way):

```bash
C:\msys64\msys2_shell.cmd -mingw64 -defterm -no-start -here -use-full-path
pacman -Sy --needed mingw-w64-x86_64-gcc mingw-w64-x86_64-vulkan-devel mingw-w64-x86_64-cmake mingw-w64-x86_64-shaderc
export CGO_ENABLED=1
go build -o sonara.exe ./cmd/sonara/
```

## Bumping whisper.cpp

1. Update the commit hash in `.whisper.cpp-commit`
2. Run `uv run scripts/fetch-headers.py` and commit the updated headers
3. Trigger the `Build whisper.cpp libs` workflow (or run `uv run scripts/build-libs.py --upload` locally)

## Releasing binaries

`Release Sonara` workflow builds and uploads `cmd/sonara` binaries for:
- Linux: `amd64`, `arm64`
- macOS: Apple Silicon and Intel
- Windows: `amd64`

It also injects the CLI version at build time via:

```bash
go build -ldflags "-X main.version=<tag>"
```

You can run releases in two ways:

1. Push a tag like `v0.1.0` (workflow trigger: `push tags: v*`)
2. Manual dispatch with input `version` (example: `v0.1.0`)
