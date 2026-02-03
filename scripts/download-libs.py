# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "httpx==0.28.1",
# ]
# ///

"""Download prebuilt whisper.cpp static libs from GitHub releases for the current platform."""

import platform, tarfile, io
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
GITHUB_REPO = "thewh1teagle/sonara"


def get_commit() -> str:
    return (ROOT / ".whisper.cpp-commit").read_text().strip()


def platform_id() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    return f"{system}-{machine}"


def download_and_extract(commit: str, plat: str):
    tag = f"libraries-{commit[:7]}"
    filename = f"whisper-libs-{plat}.tar.gz"
    url = f"https://github.com/{GITHUB_REPO}/releases/download/{tag}/{filename}"

    print(f"downloading {url}")
    resp = httpx.get(url, follow_redirects=True, timeout=120)
    resp.raise_for_status()

    out_dir = ROOT / "third_party"
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
        tar.extractall(path=out_dir)

    print(f"extracted to {out_dir} ({len(resp.content) // 1024} KB)")


def main():
    commit = get_commit()
    plat = platform_id()
    print(f"commit: {commit}")
    print(f"platform: {plat}")
    download_and_extract(commit, plat)


if __name__ == "__main__":
    main()
