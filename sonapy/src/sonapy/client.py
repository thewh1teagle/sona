from __future__ import annotations

import json
from pathlib import Path
from typing import Generator

import httpx


class Client:
    """HTTP client for the Sona API."""

    def __init__(self, base_url: str) -> None:
        self._http = httpx.Client(base_url=base_url, timeout=None)

    def close(self) -> None:
        self._http.close()

    # -- health --

    def health(self) -> dict:
        return self._http.get("/health").json()

    def ready(self) -> dict:
        return self._http.get("/ready").json()

    # -- model management --

    def load_model(self, path: str) -> dict:
        return self._http.post("/v1/models/load", json={"path": path}).json()

    def unload_model(self) -> dict:
        return self._http.delete("/v1/models").json()

    def models(self) -> dict:
        return self._http.get("/v1/models").json()

    # -- transcription --

    def transcribe(
        self,
        file_path: str | Path,
        *,
        response_format: str = "json",
        language: str = "",
        stream: bool = False,
        diarize_model: str = "",
    ) -> dict | str | Generator[dict, None, None]:
        """Transcribe an audio file.

        Returns a dict for json/verbose_json, a string for text/srt/vtt,
        or a generator of ndjson dicts when *stream* is True.

        If *diarize_model* is set (path to a Sortformer .onnx model),
        segments in verbose_json will include a ``speaker`` field.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"audio file not found: {path}")

        data: dict[str, str] = {"response_format": response_format}
        if language:
            data["language"] = language
        if stream:
            data["stream"] = "true"
        if diarize_model:
            data["diarize_model"] = diarize_model

        if stream:
            return self._stream_transcribe(path, data)

        with open(path, "rb") as f:
            r = self._http.post(
                "/v1/audio/transcriptions",
                files={"file": (path.name, f, "application/octet-stream")},
                data=data,
            )

        if response_format in ("text", "srt", "vtt"):
            return r.text
        return r.json()

    def _stream_transcribe(self, path: Path, data: dict) -> Generator[dict, None, None]:
        with open(path, "rb") as f:
            with self._http.stream(
                "POST",
                "/v1/audio/transcriptions",
                files={"file": (path.name, f, "application/octet-stream")},
                data=data,
            ) as resp:
                for line in resp.iter_lines():
                    if line:
                        yield json.loads(line)
