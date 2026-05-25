"""Ollama compatibility helpers shared by Kiron services.

The module intentionally stays small in V1: it normalizes the Ollama
response shapes Kiron relies on and provides request-body helpers used by
the existing proxy implementations. It is not a second proxy layer.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

OLLAMA_BODY_PATHS = frozenset({"/api/chat", "/api/generate"})
UNLOAD_VERIFY_TIMEOUT_S = 30.0
UNLOAD_VERIFY_POLL_S = 1.0


@dataclass(frozen=True)
class CompatResult:
    ok: bool
    severity: str = "info"
    code: str = "ok"
    message: str = ""
    data: Any = None


@dataclass(frozen=True)
class OllamaCapabilities:
    version_endpoint: CompatResult = field(default_factory=lambda: CompatResult(True))
    tags_shape: CompatResult = field(default_factory=lambda: CompatResult(True))
    ps_shape: CompatResult = field(default_factory=lambda: CompatResult(True))
    ps_size_vram: CompatResult = field(default_factory=lambda: CompatResult(True))
    generate_nonstream: CompatResult = field(default_factory=lambda: CompatResult(True))
    chat_nonstream: CompatResult = field(default_factory=lambda: CompatResult(True))
    stream_ndjson: CompatResult = field(default_factory=lambda: CompatResult(True))
    think_false: CompatResult = field(default_factory=lambda: CompatResult(True))
    keep_alive_zero_unload: CompatResult = field(default_factory=lambda: CompatResult(True))
    num_gpu_zero_chat_generate: CompatResult = field(default_factory=lambda: CompatResult(True))
    error_shape_normalizable: CompatResult = field(default_factory=lambda: CompatResult(True))


@dataclass(frozen=True)
class OllamaLoadedModel:
    name: str
    model: str
    size: int | None
    size_vram: int | None
    raw: dict[str, Any]


def canonical_model_name(name: str) -> str:
    """Add an implicit ``:latest`` tag like Ollama does internally.

    `/api/generate` accepts tagless names but `/api/ps` always lists the
    canonical name with tag, so both sides must be canonicalized before
    comparison. Only the last path segment carries the tag — a ``:`` in a
    registry host like ``localhost:5000/llama3`` is not a tag separator.
    """
    if not name:
        return name
    last = name.rsplit("/", 1)[-1]
    return name if ":" in last else f"{name}:latest"


def is_real_int(value: Any) -> bool:
    """True only for JSON integer values, not bools."""
    return isinstance(value, int) and not isinstance(value, bool)


def coerce_optional_int(value: Any) -> int | None:
    return value if is_real_int(value) else None


def normalize_loaded_model(
    raw: Any,
    *,
    require_size_vram: bool,
) -> CompatResult:
    """Normalize one `/api/ps` model entry.

    Critical VRAM paths pass ``require_size_vram=True``. In that mode a
    missing, bool, string, float or otherwise malformed ``size_vram`` is
    a hard failure. Display paths can keep rendering with ``None``.
    """
    if not isinstance(raw, dict):
        return CompatResult(False, "fail", "model_not_dict", "loaded model entry is not an object")

    name = raw.get("name")
    if not isinstance(name, str) or not name:
        return CompatResult(False, "fail", "model_name_invalid", "loaded model entry has no string name")

    model = raw.get("model")
    if not isinstance(model, str) or not model:
        model = name

    size = coerce_optional_int(raw.get("size"))
    size_vram = coerce_optional_int(raw.get("size_vram"))
    if size is not None and size < 0:
        size = None
    size_vram_negative = size_vram is not None and size_vram < 0
    if size_vram_negative:
        size_vram = None
    if require_size_vram and size_vram is None:
        if size_vram_negative:
            return CompatResult(
                False,
                "fail",
                "size_vram_negative",
                "loaded model entry has negative size_vram",
                {"name": name, "raw": raw},
            )
        return CompatResult(
            False,
            "fail",
            "size_vram_invalid",
            "loaded model entry has no real integer size_vram",
            {"name": name, "raw": raw},
        )

    return CompatResult(
        True,
        "info",
        "ok",
        "",
        OllamaLoadedModel(
            name=name,
            model=model,
            size=size,
            size_vram=size_vram,
            raw=dict(raw),
        ),
    )


def normalize_ps_response(raw: Any, *, require_size_vram: bool) -> CompatResult:
    if not isinstance(raw, dict):
        return CompatResult(False, "fail", "ps_root_invalid", "/api/ps root is not an object")
    models_raw = raw.get("models")
    if not isinstance(models_raw, list):
        return CompatResult(False, "fail", "ps_models_invalid", "/api/ps models is not a list")
    models: list[OllamaLoadedModel] = []
    failures: list[CompatResult] = []
    for item in models_raw:
        normalized = normalize_loaded_model(item, require_size_vram=require_size_vram)
        if normalized.ok:
            models.append(normalized.data)
        else:
            failures.append(normalized)
    if require_size_vram and failures:
        first = failures[0]
        return CompatResult(
            False,
            first.severity,
            first.code,
            first.message,
            {"models": models, "failures": failures, "raw": raw},
        )
    return CompatResult(
        True,
        "info" if not failures else "warn",
        "ok" if not failures else "ps_entry_warnings",
        "",
        {"models": models, "failures": failures, "raw": raw},
    )


def normalize_tags_response(raw: Any) -> CompatResult:
    if not isinstance(raw, dict):
        return CompatResult(False, "fail", "tags_root_invalid", "/api/tags root is not an object")
    models_raw = raw.get("models")
    if not isinstance(models_raw, list):
        return CompatResult(False, "fail", "tags_models_invalid", "/api/tags models is not a list")
    models: list[dict[str, Any]] = []
    failures: list[CompatResult] = []
    for item in models_raw:
        if not isinstance(item, dict):
            failures.append(CompatResult(False, "warn", "tag_entry_invalid", "tag entry is not an object"))
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name:
            failures.append(CompatResult(False, "warn", "tag_name_invalid", "tag entry has no string name"))
            continue
        copied = dict(item)
        copied["model"] = copied.get("model") if isinstance(copied.get("model"), str) else name
        models.append(copied)
    return CompatResult(
        True,
        "info" if not failures else "warn",
        "ok" if not failures else "tag_entry_warnings",
        "",
        {"models": models, "failures": failures, "raw": raw},
    )


def apply_think_false_bytes(body: bytes, path: str) -> bytes:
    """Preserve native Ollama behavior except defaulting think to false.

    Applies only to chat/generate. Explicit ``think:true`` survives.
    Invalid JSON and non-object roots are returned unchanged.
    """
    # Adapter ist Single Source of Truth fuer diese Mutation: Pfad selbst
    # normalisieren, damit /api/chat/ oder /api/chat?stream=true von Future-
    # Callern nicht versehentlich think:true zum Client durchlassen.
    path = path.split("?", 1)[0].rstrip("/")
    if path not in OLLAMA_BODY_PATHS or not body:
        return body
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return body
    if not isinstance(payload, dict):
        return body
    if payload.get("think") is True:
        return body
    payload["think"] = False
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def ensure_think_false_dict(payload: dict[str, Any], path: str) -> dict[str, Any]:
    if path in OLLAMA_BODY_PATHS and payload.get("think") is not True:
        payload["think"] = False
    return payload


def ensure_num_gpu_zero(options: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(options, dict):
        options = {}
    if "num_gpu" not in options:
        options["num_gpu"] = 0
    return options


def normalize_backend_error(exc: BaseException, backend: str = "ollama") -> CompatResult:
    if isinstance(exc, httpx.ConnectError):
        code = "connect_error"
    elif isinstance(exc, httpx.TimeoutException):
        code = "timeout"
    elif isinstance(exc, httpx.RemoteProtocolError):
        code = "remote_protocol_error"
    elif isinstance(exc, httpx.ReadError):
        code = "read_error"
    elif isinstance(exc, httpx.WriteError):
        code = "write_error"
    elif isinstance(exc, httpx.RequestError):
        code = "request_error"
    else:
        code = "exception"
    return CompatResult(
        False,
        "fail",
        code,
        f"{backend} backend error: {type(exc).__name__}: {exc}",
        {"backend": backend, "exception": type(exc).__name__},
    )


class OllamaCompatClient:
    """Small async client for Ollama contract probes.

    If ``client`` is injected, the caller keeps ownership and ``aclose`` will
    not close it. Otherwise this class owns its internal ``httpx.AsyncClient``.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11435",
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> None:
        if client is not None and timeout is not None:
            raise ValueError(
                "OllamaCompatClient: timeout must be set on the injected client, "
                "not as a separate argument"
            )
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=base_url,
            timeout=10.0 if timeout is None else timeout,
        )

    async def __aenter__(self) -> "OllamaCompatClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def version(self) -> CompatResult:
        try:
            resp = await self._client.get("/api/version")
        except httpx.RequestError as exc:
            return normalize_backend_error(exc)
        if resp.status_code != 200:
            return CompatResult(False, "fail", "version_http", f"/api/version HTTP {resp.status_code}")
        try:
            data = resp.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            return CompatResult(False, "fail", "version_decode", f"/api/version body not JSON: {exc}")
        version = data.get("version") if isinstance(data, dict) else None
        if not isinstance(version, str) or not version:
            return CompatResult(False, "fail", "version_shape", "/api/version has no string version", data)
        return CompatResult(True, "info", "ok", "", data)

    async def tags(self) -> CompatResult:
        try:
            resp = await self._client.get("/api/tags")
        except httpx.RequestError as exc:
            return normalize_backend_error(exc)
        if resp.status_code != 200:
            return CompatResult(False, "fail", "tags_http", f"/api/tags HTTP {resp.status_code}")
        try:
            payload = resp.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            return CompatResult(False, "fail", "tags_decode", f"/api/tags body not JSON: {exc}")
        return normalize_tags_response(payload)

    async def ps(self, *, require_size_vram: bool = False) -> CompatResult:
        try:
            resp = await self._client.get("/api/ps")
        except httpx.RequestError as exc:
            return normalize_backend_error(exc)
        if resp.status_code != 200:
            return CompatResult(False, "fail", "ps_http", f"/api/ps HTTP {resp.status_code}")
        try:
            payload = resp.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            return CompatResult(False, "fail", "ps_decode", f"/api/ps body not JSON: {exc}")
        return normalize_ps_response(payload, require_size_vram=require_size_vram)

    async def unload(self, model: str) -> CompatResult:
        try:
            resp = await self._client.post(
                "/api/generate",
                json={"model": model, "keep_alive": 0, "stream": False},
                timeout=UNLOAD_VERIFY_TIMEOUT_S,
            )
        except httpx.RequestError as exc:
            return normalize_backend_error(exc)
        if not (200 <= resp.status_code < 300):
            return CompatResult(False, "fail", "unload_http", f"unload HTTP {resp.status_code}")
        try:
            payload = resp.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            return CompatResult(False, "fail", "unload_decode", f"unload body not JSON: {exc}")
        # Ollama can return 200 with an error body (e.g. unknown model arg);
        # short-circuit instead of waiting on the 30s verify loop.
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, str) and error:
                return CompatResult(
                    False,
                    "fail",
                    "unload_rejected",
                    f"unload rejected by backend: {error}",
                    {"model": model, "error": error},
                )
        deadline = time.monotonic() + UNLOAD_VERIFY_TIMEOUT_S
        canonical = canonical_model_name(model)
        while True:
            ps_result = await self.ps(require_size_vram=False)
            deadline_reached = time.monotonic() >= deadline
            if not ps_result.ok:
                if deadline_reached:
                    return CompatResult(
                        False,
                        "fail",
                        "unload_verify_ps_failed",
                        "unload verification /api/ps failed",
                        {
                            "model": model,
                            "ps_code": ps_result.code,
                            "ps_message": ps_result.message,
                        },
                    )
                await asyncio.sleep(
                    min(UNLOAD_VERIFY_POLL_S, max(0.0, deadline - time.monotonic()))
                )
                continue
            loaded = ps_result.data["models"]
            ps_failures = ps_result.data.get("failures") or []
            target_loaded = any(
                canonical_model_name(item.name) == canonical
                or canonical_model_name(item.model) == canonical
                for item in loaded
            )
            if not target_loaded and not ps_failures:
                return CompatResult(True, "info", "ok", "unload verified", {"model": model})
            if deadline_reached:
                if target_loaded:
                    return CompatResult(
                        False,
                        "fail",
                        "unload_verify_timeout",
                        "model still listed in /api/ps after unload",
                        {"model": model, "loaded": [item.name for item in loaded]},
                    )
                return CompatResult(
                    False,
                    "fail",
                    "unload_verify_ps_failed",
                    "unload verification /api/ps returned malformed entries",
                    {
                        "model": model,
                        "loaded": [item.name for item in loaded],
                        "ps_failures": [
                            {"code": f.code, "message": f.message} for f in ps_failures
                        ],
                    },
                )
            await asyncio.sleep(
                min(UNLOAD_VERIFY_POLL_S, max(0.0, deadline - time.monotonic()))
            )
