# Vulkan Crash Debug — Status Report

## Problem

`sonara.exe` segfaults (`PC=0x0`, null function pointer) during `whisper_full` when using the Vulkan backend on an AMD Radeon iGPU in MSYS2.

## Environment

- GPU: AMD Radeon(TM) Graphics (iGPU, `matrix cores: none`, `int dot: 0`)
- OS: Windows, MSYS2 MINGW64 shell
- whisper.cpp commit: `aa1bc0d1a6dfd70dbb9f60c11df12441e03a9075`
- Go linker: `go build` with cgo, uses `gcc` as external linker

## What works

| Test | Result |
|------|--------|
| whisper-cli (cmake-built) + Vulkan + flash_attn=1 | **OK** |
| whisper-cli (cmake-built) + Vulkan + flash_attn=0 | **OK** |
| whisper-cli (cmake-built) + Vulkan + greedy (bs=1 bo=1) | **OK** |
| whisper-cli (cmake-built) + Vulkan + greedy + no flash attn | **OK** |
| C test (`test_whisper.c`) linked with cmake-matching flags (`c++`, no static runtime, correct link order, `-lmingwthrd`, Windows system libs) | **OK** |
| sonara CPU-only | **OK** |

## What crashes

| Test | Result |
|------|--------|
| C test linked with original sonara flags (`gcc -static-libstdc++ -static-libgcc`, missing `-lmingwthrd`, missing Windows libs) | **CRASH** |
| sonara (Go/cgo) with original LDFLAGS | **CRASH** |
| sonara (Go/cgo) with cmake-matching LDFLAGS (added `-lmingwthrd`, Windows libs, removed `-static-libstdc++ -static-libgcc`) | **CRASH** |
| sonara (Go/cgo) with `-buildmode=exe` (non-PIE) | **CRASH** |
| sonara (Go/cgo) with `CreateThread` wrapper (run whisper_full on native thread) | **CRASH** |
| sonara (Go/cgo) with `flash_attn=true` | **CRASH** |

## Key findings

1. **The C test crash was fixed by link flags** — specifically: using `c++` linker, adding `-lmingwthrd`, Windows system libs, correct link order, and removing `-static-libstdc++ -static-libgcc`. The critical factor is likely `-static-libstdc++ -static-libgcc` vs dynamic, or `-lmingwthrd`.

2. **Go/cgo still crashes even with the same link flags.** This means Go's runtime itself causes the issue, not just the link flags. Go's process-wide vectored exception handler (VEH) and/or its thread management likely interfere with the Vulkan AMD driver.

3. **The crash is at `PC=0x0`** — a null function pointer call. This happens inside the Vulkan compute path during `whisper_full`. The AMD driver or ggml-vulkan dispatches through a function pointer that is null.

## Not yet tried

- [ ] Isolate which specific C link flag fixes the C test crash (`-static-libstdc++`? `-lmingwthrd`? link order?)
- [ ] Build whisper as a shared DLL (`.dll`) and have Go load it via `syscall.LoadDLL` instead of static linking — this changes how the linker resolves symbols
- [ ] Use Go's `-linkmode=external` with explicit `-extldflags` matching the cmake link command exactly
- [ ] Try `go build -ldflags="-extldflags '-Wl,--stack,8388608'"` for larger thread stack
- [ ] Try removing Go's VEH at process init via a C init function that calls `RemoveVectoredExceptionHandler`
- [ ] Check if the issue is Go's linker generating a broken executable by dumping and comparing the import tables of the working C test vs the Go binary
- [ ] Try building with CGO using `c++` as the external linker: `CGO_LDFLAGS` or `CC=c++`
- [ ] Try `GOEXPERIMENT=nocgo` or alternative Go FFI approaches
- [ ] File an issue on Go or whisper.cpp with repro steps

## Files modified

- `internal/whisper/whisper_windows.go` — updated LDFLAGS (link order, added `-lmingwthrd`, Windows libs, removed `-static-libstdc++ -static-libgcc`), added `CreateThread` wrapper (still crashes)
- `BUILDING.md` — added `mingw-w64-x86_64-shaderc` to MSYS2 package list
- `test_whisper.c` — standalone C test for reproducing the crash outside Go

## Files created (temporary, for debugging)

- `test_whisper.c` — minimal C reproduction
- `test_whisper.exe` — compiled test
- `whisper-src/` — cloned whisper.cpp source
- `whisper-build/` — local cmake build (whisper-cli works here)
