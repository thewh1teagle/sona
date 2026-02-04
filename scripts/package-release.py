# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "httpx==0.28.1",
# ]
# ///

"""Package a Sonara release archive with a bundled ffmpeg binary."""

import argparse
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path

import httpx


FFMPEG_URLS = {
    ("darwin", "amd64"): "https://github.com/eugeneware/ffmpeg-static/releases/download/b6.1.1/ffmpeg-darwin-x64",
    ("darwin", "arm64"): "https://github.com/eugeneware/ffmpeg-static/releases/download/b6.1.1/ffmpeg-darwin-arm64",
    ("linux", "amd64"): "https://github.com/eugeneware/ffmpeg-static/releases/download/b6.1.1/ffmpeg-linux-x64",
    ("linux", "arm64"): "https://github.com/eugeneware/ffmpeg-static/releases/download/b6.1.1/ffmpeg-linux-arm64",
    ("windows", "amd64"): "https://github.com/eugeneware/ffmpeg-static/releases/download/b6.1.1/ffmpeg-win32-x64",
}


def download_bytes(url: str) -> bytes:
    print(f"downloading {url}")
    resp = httpx.get(url, follow_redirects=True, timeout=180)
    resp.raise_for_status()
    return resp.content


def resolve_ffmpeg_url(goos: str, goarch: str) -> str:
    key = (goos, goarch)
    if key not in FFMPEG_URLS:
        raise KeyError(f"unsupported target for ffmpeg URL: {goos}/{goarch}")
    return FFMPEG_URLS[key]


def copy_ffmpeg(stage_dir: Path, goos: str, goarch: str):
    ffmpeg_name = "ffmpeg.exe" if goos == "windows" else "ffmpeg"
    ffmpeg_path = stage_dir / ffmpeg_name
    data = download_bytes(resolve_ffmpeg_url(goos, goarch))
    ffmpeg_path.write_bytes(data)
    ffmpeg_path.chmod(0o755)


def package(binary_path: Path, out_path: Path, goos: str, goarch: str):
    if not binary_path.exists():
        raise FileNotFoundError(binary_path)

    with tempfile.TemporaryDirectory(prefix="sonara-package-") as td:
        stage_dir = Path(td) / "bundle"
        stage_dir.mkdir(parents=True, exist_ok=True)

        binary_name = "sonara.exe" if goos == "windows" else "sonara"
        target_binary = stage_dir / binary_name
        shutil.copy2(binary_path, target_binary)
        target_binary.chmod(0o755)

        copy_ffmpeg(stage_dir, goos, goarch)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        if goos == "windows":
            with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for item in sorted(stage_dir.iterdir()):
                    zf.write(item, arcname=item.name)
        else:
            with tarfile.open(out_path, "w:gz") as tar:
                for item in sorted(stage_dir.iterdir()):
                    tar.add(item, arcname=item.name)

    print(f"packaged {out_path} ({out_path.stat().st_size // 1024} KB)")


def main():
    parser = argparse.ArgumentParser(description="Package Sonara release archive")
    parser.add_argument("--binary", required=True, help="Built sonara binary path")
    parser.add_argument("--goos", required=True, choices=["linux", "darwin", "windows"])
    parser.add_argument("--goarch", required=True, choices=["amd64", "arm64"])
    parser.add_argument("--out", required=True, help="Output archive path (.tar.gz or .zip)")
    args = parser.parse_args()

    package(Path(args.binary), Path(args.out), args.goos, args.goarch)


if __name__ == "__main__":
    main()
