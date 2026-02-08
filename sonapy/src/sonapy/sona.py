from __future__ import annotations

import atexit
from pathlib import Path
from typing import Generator

from sonapy.client import Client
from sonapy.runner import Runner, SonaError


class Sona:
    """Spawn a Sona runner and talk to it.

    Usage::

        with Sona() as sona:
            sona.load_model("model.bin")
            print(sona.transcribe("audio.wav"))
    """

    def __init__(self, port: int = 0) -> None:
        self._runner = Runner(port=port)
        self.port: int = self._runner.port
        self.base_url: str = f"http://localhost:{self.port}"
        self._client = Client(base_url=self.base_url)
        atexit.register(self.stop)

    def __enter__(self) -> Sona:
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    # -- lifecycle --

    def stop(self) -> None:
        """Shut down the runner."""
        self._client.close()
        self._runner.stop()

    @property
    def alive(self) -> bool:
        return self._runner.alive

    # -- delegates to client --

    def health(self) -> dict:
        return self._client.health()

    def ready(self) -> dict:
        return self._client.ready()

    def load_model(self, path: str) -> dict:
        return self._client.load_model(path)

    def unload_model(self) -> dict:
        return self._client.unload_model()

    def models(self) -> dict:
        return self._client.models()

    def transcribe(
        self,
        file_path: str | Path,
        *,
        response_format: str = "json",
        language: str = "",
        stream: bool = False,
        diarize_model: str = "",
    ) -> dict | str | Generator[dict, None, None]:
        return self._client.transcribe(
            file_path,
            response_format=response_format,
            language=language,
            stream=stream,
            diarize_model=diarize_model,
        )
