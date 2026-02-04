# Vulkan Crash Root Cause Found

## Root cause

`gcc` implicitly links `-lgcc_eh` (static exception handling), while `c++` implicitly links `-lgcc_s` (shared libgcc with SEH support). The C++ code in ggml-vulkan requires proper Windows SEH (Structured Exception Handling) initialization provided by `libgcc_s_seh-1.dll`. Without it, a function pointer in the Vulkan dispatch path is null, causing the `PC=0x0` crash.

## Proof

Same source (`test_whisper.c`), same flags, same static libs:

```
gcc ... -lstdc++             → CRASH (links -lgcc_eh by default)
gcc ... -lstdc++ -lgcc_s     → WORKS (overrides with shared SEH runtime)
c++ ...                      → WORKS (implicitly links -lgcc_s)
```

## Fix for sonara

In `internal/whisper/whisper_windows.go`, the current LDFLAGS are:

```
#cgo LDFLAGS: -lvulkan-1 -lstdc++ -lm -lpthread -lgomp
#cgo LDFLAGS: -static-libstdc++ -static-libgcc
```

`-static-libgcc` forces the static `-lgcc_eh`. This must be removed and `-lgcc_s` must be added explicitly. The fix:

```
#cgo LDFLAGS: -lvulkan-1 -lstdc++ -lm -lpthread -lgomp -lgcc_s
#cgo LDFLAGS: -static-libstdc++
```

This keeps libstdc++ static (no extra DLL dependency) but uses the shared libgcc (`libgcc_s_seh-1.dll`) which provides proper SEH unwinding that the Vulkan backend needs.

**Trade-off:** `libgcc_s_seh-1.dll` must be present at runtime. On MSYS2 it's in `/mingw64/bin/`. For distribution, it needs to be shipped alongside `sonara.exe` or the user needs MSYS2's mingw64 bin on PATH.

## Not yet verified

- [ ] Apply the LDFLAGS fix to the Go build and confirm sonara works with Vulkan
- [ ] Confirm the CI build also needs this change
- [ ] Decide whether to ship `libgcc_s_seh-1.dll` or document the runtime dependency
