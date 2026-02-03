# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "httpx==0.28.1",
# ]
# ///

"""Fetch whisper.h and ggml headers from GitHub into third_party/include/. Each file is stamped with the fetch date and source commit."""

import httpx, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WHISPER_REPO = "ggml-org/whisper.cpp"
COMMIT = (ROOT / ".whisper.cpp-commit").read_text().strip()

HEADERS = [
    (WHISPER_REPO, "include/whisper.h"),
    (WHISPER_REPO, "ggml/include/ggml.h"),
    (WHISPER_REPO, "ggml/include/ggml-cpu.h"),
    (WHISPER_REPO, "ggml/include/ggml-alloc.h"),
    (WHISPER_REPO, "ggml/include/ggml-backend.h"),
]

out_dir = ROOT / "third_party/include"
out_dir.mkdir(parents=True, exist_ok=True)
now = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

for repo, path in HEADERS:
    name = Path(path).name
    content = httpx.get(f"https://raw.githubusercontent.com/{repo}/{COMMIT}/{path}").text
    (out_dir / name).write_text(
        f"// Fetched: {now}\n"
        f"// Source: https://github.com/{repo}/blob/{COMMIT}/{path}\n"
        f"// Commit: {COMMIT}\n\n"
        + content
    )
    print(f"wrote {name} (commit {COMMIT})")
