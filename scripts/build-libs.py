# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

"""Build whisper.cpp static libs for the current platform and optionally upload to a GitHub release."""

import argparse, os, platform, shutil, subprocess, tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
WHISPER_REPO = "https://github.com/ggml-org/whisper.cpp.git"


def run(*args: str, cwd: str | None = None):
    print(f"$ {' '.join(args)}")
    subprocess.run(args, cwd=cwd, check=True)


def get_whisper_commit() -> str:
    return (ROOT / ".whisper.cpp-commit").read_text().strip()


def platform_id() -> str:
    system = platform.system().lower()  # darwin, linux, windows
    machine = platform.machine().lower()  # arm64, x86_64, amd64
    return f"{system}-{machine}"


def clone(commit: str, src_dir: Path):
    if src_dir.exists():
        shutil.rmtree(src_dir)
    src_dir.mkdir(parents=True)
    run("git", "init", cwd=str(src_dir))
    run("git", "remote", "add", "origin", WHISPER_REPO, cwd=str(src_dir))
    run("git", "fetch", "--depth", "1", "origin", commit, cwd=str(src_dir))
    run("git", "checkout", "FETCH_HEAD", cwd=str(src_dir))
    run(
        "git",
        "submodule",
        "update",
        "--init",
        "--depth",
        "1",
        "--recursive",
        cwd=str(src_dir),
    )


def cmake_flags() -> list[str]:
    flags = [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=OFF",
        "-DWHISPER_BUILD_EXAMPLES=OFF",
        "-DWHISPER_BUILD_TESTS=OFF",
        "-DWHISPER_BUILD_SERVER=OFF",
    ]
    system = platform.system()
    if system == "Darwin":
        flags += ["-DGGML_METAL=ON", "-DGGML_METAL_EMBED_LIBRARY=ON"]
    elif system in ("Linux", "Windows"):
        flags += ["-DGGML_VULKAN=ON"]
    if system == "Windows":
        flags += ["-G", "MinGW Makefiles"]
    return flags


def build(src_dir: Path, build_dir: Path):
    if build_dir.exists():
        shutil.rmtree(build_dir)
    run("cmake", "-S", str(src_dir), "-B", str(build_dir), *cmake_flags())
    jobs = str(os.cpu_count() or 4)
    run("cmake", "--build", str(build_dir), "--config", "Release", f"-j{jobs}")


def lib_names() -> list[str]:
    common = ["libwhisper.a", "libggml.a", "libggml-base.a", "libggml-cpu.a"]
    system = platform.system()
    if system == "Darwin":
        common += ["libggml-metal.a", "libggml-blas.a"]
    elif system in ("Linux", "Windows"):
        common += ["libggml-vulkan.a"]
    return common


def find_lib(build_dir: Path, name: str) -> tuple[Path, str]:
    """Find a static lib by expected name (and Windows no-prefix fallback)."""
    candidates = [name]
    if platform.system() == "Windows" and name.startswith("lib"):
        candidates.append(name[3:])

    for candidate in candidates:
        for p in build_dir.rglob(candidate):
            return p, candidate
    raise FileNotFoundError(f"{name} not found under {build_dir}")


def package(build_dir: Path, src_dir: Path, archive: Path):
    pkg = build_dir / "pkg"
    if pkg.exists():
        shutil.rmtree(pkg)
    (pkg / "lib").mkdir(parents=True)
    (pkg / "include").mkdir(parents=True)
    for name in lib_names():
        lib, matched_name = find_lib(build_dir, name)
        print(f"  found {matched_name} at {lib} (packaged as {name})")
        # Normalize archive names so cgo -l<name> works consistently on every platform.
        shutil.copy2(lib, pkg / "lib" / name)
    shutil.copy2(src_dir / "include/whisper.h", pkg / "include/whisper.h")
    with tarfile.open(archive, "w:gz") as tar:
        for item in pkg.iterdir():
            tar.add(item, arcname=item.name)
    print(f"packaged {archive} ({archive.stat().st_size // 1024} KB)")


def upload(archive: Path, tag: str):
    import time

    # Release may already exist from another matrix job — ignore error
    subprocess.run(["gh", "release", "create", tag, "--generate-notes"], check=False)
    # Retry upload — release may take a moment to become available
    for attempt in range(5):
        result = subprocess.run(
            ["gh", "release", "upload", tag, str(archive), "--clobber"]
        )
        if result.returncode == 0:
            break
        print(f"upload attempt {attempt + 1} failed, retrying...")
        time.sleep(3)
    else:
        raise RuntimeError(f"failed to upload {archive.name} after 5 attempts")
    print(f"uploaded {archive.name} to release {tag}")


def main():
    parser = argparse.ArgumentParser(description="Build whisper.cpp static libs")
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload to GitHub release (tag derived from commit)",
    )
    args = parser.parse_args()

    commit = get_whisper_commit()
    plat = platform_id()
    src_dir = ROOT / "whisper-src"
    build_dir = ROOT / "whisper-build"
    archive = ROOT / f"whisper-libs-{plat}.tar.gz"

    print(f"commit: {commit}")
    print(f"platform: {plat}")

    clone(commit, src_dir)
    build(src_dir, build_dir)
    package(build_dir, src_dir, archive)

    if args.upload:
        tag = f"libraries-{commit[:7]}"
        upload(archive, tag)


if __name__ == "__main__":
    main()
