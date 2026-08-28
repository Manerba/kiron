"""Fixed-loopback adapter for KIron's already running local Ollama."""

from __future__ import annotations

import httpx


DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11435"


class LocalOllamaAPI:
    """Expose only local inventory operations required for registration."""

    __slots__ = ()

    @staticmethod
    def _request(method: str, path: str, **kwargs: object) -> object:
        with httpx.Client(
            base_url=DEFAULT_OLLAMA_BASE_URL,
            timeout=10.0,
            trust_env=False,
        ) as client:
            response = client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json()

    def list_models(self) -> object:
        return self._request("GET", "/api/tags")

    def show_model(self, name: str) -> object:
        return self._request("POST", "/api/show", json={"model": name})


__all__ = ["DEFAULT_OLLAMA_BASE_URL", "LocalOllamaAPI"]
