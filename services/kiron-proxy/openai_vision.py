"""Bounded data-image decoding, without network or backend access.

Images are decoded in a disposable process, never a cancellation-resistant
thread. EXIF orientation is applied without resizing; transparency is composed
onto white. WebP and changed pixels are encoded as PNG. Other PNG/JPEG files
remain byte-identical. Original format/digest remain explicit in ImagePart.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import struct
import sys
import threading
import time
from types import MappingProxyType

MAX_IMAGES = 4
MAX_SOURCE_BYTES = 2 * 1024**2
MAX_PIXELS = 16_000_000
MAX_TOTAL_PIXELS = 32_000_000
MAX_SIDE = 8192
MAX_NORMALIZED_BYTES = 64 * 1024**2
DECODE_SECONDS = 2.0
WORKER_MEMORY = 512 * 1024**2
PILLOW_VERSION = "12.3.0"
FORMATS = {"image/png": "PNG", "image/jpeg": "JPEG", "image/webp": "WEBP"}
_WORKERS = threading.BoundedSemaphore(2)


class VisionError(ValueError):
    def __init__(self, code="invalid_request", param=None, status=400):
        self.code, self.param, self.status = code, param, status
        super().__init__("Image input could not be validated.")


@dataclass(frozen=True, slots=True)
class PendingImage:
    media_type: str
    data: bytes
    width: int
    height: int
    detail: str
    param: str


@dataclass(slots=True)
class ImageBudget:
    images: int = 0
    source_bytes: int = 0
    pixels: int = 0

    def add(self, size, width, height, param):
        self.images += 1
        self.source_bytes += size
        self.pixels += width * height
        if (self.images > MAX_IMAGES or self.source_bytes > MAX_SOURCE_BYTES
                or self.pixels > MAX_TOTAL_PIXELS):
            raise VisionError(param=param)


def _dimensions(data, mime):
    """Read bounded container metadata before an image library allocates pixels."""
    if mime == "image/png":
        if len(data) < 33 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[8:16] != b"\0\0\0\rIHDR":
            raise ValueError("PNG header")
        width, height = struct.unpack(">II", data[16:24])
        position, ended = 8, False
        while position + 12 <= len(data):
            length = int.from_bytes(data[position:position + 4], "big")
            kind = data[position + 4:position + 8]
            position += length + 12
            if position > len(data) or kind == b"acTL":
                raise ValueError("PNG container or animation")
            if kind == b"IEND":
                ended = length == 0 and position == len(data)
                break
        if not ended:
            raise ValueError("PNG end")
    elif mime == "image/jpeg":
        if not data.startswith(b"\xff\xd8") or not data.endswith(b"\xff\xd9"):
            raise ValueError("JPEG header/end")
        position, width, height = 2, 0, 0
        while position < len(data):
            if data[position] != 0xFF:
                raise ValueError("JPEG marker")
            while position < len(data) and data[position] == 0xFF:
                position += 1
            if position >= len(data):
                break
            marker = data[position]
            position += 1
            if marker in (0xDA, 0xD9):
                break
            if marker == 0x01 or 0xD0 <= marker <= 0xD7:
                continue
            if position + 2 > len(data):
                raise ValueError("JPEG segment")
            length = int.from_bytes(data[position:position + 2], "big")
            if length < 2 or position + length > len(data):
                raise ValueError("JPEG segment")
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                if length < 8:
                    raise ValueError("JPEG dimensions")
                height, width = struct.unpack(">HH", data[position + 3:position + 7])
                break
            position += length
    elif mime == "image/webp":
        if (len(data) < 20 or data[:4] != b"RIFF" or data[8:12] != b"WEBP"
                or int.from_bytes(data[4:8], "little") + 8 != len(data)):
            raise ValueError("WebP container")
        position, width, height = 12, 0, 0
        while position + 8 <= len(data):
            kind = data[position:position + 4]
            length = int.from_bytes(data[position + 4:position + 8], "little")
            start = position + 8
            position = start + length + (length & 1)
            if position > len(data) or kind in (b"ANIM", b"ANMF"):
                raise ValueError("WebP container or animation")
            if kind == b"VP8X":
                if length != 10 or data[start] & 0x02:
                    raise ValueError("WebP extended header or animation")
                width = 1 + int.from_bytes(data[start + 4:start + 7], "little")
                height = 1 + int.from_bytes(data[start + 7:start + 10], "little")
            elif kind == b"VP8L" and not width:
                if length < 5 or data[start] != 0x2F:
                    raise ValueError("WebP lossless header")
                bits = int.from_bytes(data[start + 1:start + 5], "little")
                width, height = (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
            elif kind == b"VP8 " and not width:
                if length < 10 or data[start] & 1 or data[start + 3:start + 6] != b"\x9d\x01\x2a":
                    raise ValueError("WebP frame header")
                width = int.from_bytes(data[start + 6:start + 8], "little") & 0x3FFF
                height = int.from_bytes(data[start + 8:start + 10], "little") & 0x3FFF
        if position != len(data):
            raise ValueError("WebP trailing data")
    else:
        raise ValueError("format")
    if not (0 < width <= MAX_SIDE and 0 < height <= MAX_SIDE and width * height <= MAX_PIXELS):
        raise ValueError("image dimensions")
    return width, height


def parse_image_url(value, *, param, budget):
    if type(value) is not dict or "url" not in value:
        raise VisionError(param=param)
    if value.keys() - {"url", "detail"}:
        raise VisionError("unsupported_parameter", param)
    url, detail = value["url"], value.get("detail", "auto")
    if type(url) is not str or not url or type(detail) is not str or detail not in ("auto", "low", "high"):
        raise VisionError(param=param)
    if not url.startswith("data:"):
        raise VisionError("unsupported_capability", param + ".url")
    prefix, separator, encoded = url.partition(",")
    expected = {f"data:{mime};base64": mime for mime in FORMATS}
    if prefix not in expected or not separator or not encoded:
        raise VisionError(param=param + ".url")
    # Before allocating decoded bytes, bound both this image and the remaining
    # request budget. Canonical Base64 also rejects whitespace/excess padding.
    remaining = MAX_SOURCE_BYTES - budget.source_bytes
    if budget.images >= MAX_IMAGES or len(encoded) > 4 * ((remaining + 2) // 3):
        raise VisionError(param=param)
    try:
        data = base64.b64decode(encoded, validate=True)
        if not data or len(data) > remaining or base64.b64encode(data).decode("ascii") != encoded:
            raise ValueError("noncanonical base64")
        width, height = _dimensions(data, expected[prefix])
    except (ValueError, binascii.Error):
        raise VisionError(param=param + ".url") from None
    budget.add(len(data), width, height, param)
    return PendingImage(expected[prefix], data, width, height, detail, param)


def collect_chat_images(data):
    """Image-only structural preflight; the closed chat parser owns other fields."""
    result, budget = {}, ImageBudget()
    if type(data) is not dict or type(data.get("messages")) is not list:
        return result
    if not 1 <= len(data["messages"]) <= 512:
        raise VisionError(param="messages")
    for mi, message in enumerate(data["messages"]):
        if type(message) is not dict or type(message.get("content")) is not list:
            continue
        content = message["content"]
        if not 1 <= len(content) <= 64:
            raise VisionError(param=f"messages[{mi}].content")
        for pi, part in enumerate(content):
            if type(part) is not dict or part.get("type") != "image_url":
                continue
            path = f"messages[{mi}].content[{pi}]"
            if message.get("role") != "user":
                raise VisionError(param=path)
            if part.keys() - {"type", "image_url"}:
                raise VisionError("unsupported_parameter", path)
            result[mi, pi] = parse_image_url(part.get("image_url"), param=path + ".image_url", budget=budget)
    return result


def _remaining(context, deadline):
    if context.cancellation.is_set():
        raise VisionError("resource_busy", status=503)
    remaining = min(deadline, context.deadline_monotonic) - time.monotonic()
    if remaining <= 0:
        raise VisionError("timeout", status=504)
    return remaining


async def _read_bounded(stream, limit):
    value = bytearray()
    while block := await stream.read(65536):
        if len(value) + len(block) > limit:
            raise VisionError("unsupported_value")
        value.extend(block)
    return bytes(value)


async def decode_image(pending, context):
    deadline = time.monotonic() + DECODE_SECONDS
    _remaining(context, deadline)
    if not _WORKERS.acquire(blocking=False):
        raise VisionError("overloaded", pending.param, 429)
    process, tasks = None, []
    try:
        process = await asyncio.create_subprocess_exec(sys.executable, "-I", "-B", str(Path(__file__).resolve()),
            "--decode", pending.media_type, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL, start_new_session=True,
            env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"})

        async def send():
            process.stdin.write(pending.data)
            await process.stdin.drain()
            process.stdin.close()

        tasks = [asyncio.create_task(send()), asyncio.create_task(_read_bounded(process.stdout, MAX_NORMALIZED_BYTES + 4096)),
                 asyncio.create_task(process.wait())]
        while not all(task.done() for task in tasks):
            await asyncio.wait(tasks, timeout=min(.025, _remaining(context, deadline)), return_when=asyncio.FIRST_EXCEPTION)
            for task in tasks:
                if task.done() and not task.cancelled() and task.exception() is not None:
                    raise task.exception()
        _remaining(context, deadline)
        output = tasks[1].result()
        if process.returncode == 2:
            raise VisionError("unsupported_value", pending.param)
        if process.returncode == 3:
            raise VisionError("model_unavailable", pending.param, 503)
        if process.returncode != 0:
            raise VisionError(param=pending.param)
        header, separator, payload = output.partition(b"\n")
        metadata = json.loads(header)
        if (not separator or len(header) > 4096 or metadata["media_type"] not in ("image/png", "image/jpeg")
                or type(metadata["width"]) is not int or type(metadata["height"]) is not int
                or _dimensions(payload, metadata["media_type"]) != (metadata["width"], metadata["height"])):
            raise VisionError(param=pending.param)
        from kiron_common.local_inference import ImagePart
        return ImagePart(metadata["media_type"], payload, hashlib.sha256(payload).hexdigest(),
                         metadata["width"], metadata["height"], detail=pending.detail,
                         source_media_type=pending.media_type, source_sha256=hashlib.sha256(pending.data).hexdigest())
    except (OSError, ValueError, KeyError, TypeError) as error:
        if isinstance(error, VisionError):
            raise
        raise VisionError(param=pending.param) from None
    finally:
        # This fixed decoder never launches other programs. Killing the direct
        # owned process is sufficient; do not signal a potentially reused PGID.
        try:
            if process is not None and process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            if process is not None:
                # A reader that rejected excess output may leave stdout paused
                # at its buffer limit. Reap waits for pipe EOF as well as exit;
                # discard the remaining pipe bytes without retaining them.
                async def discard():
                    while await process.stdout.read(65536):
                        pass
                await asyncio.wait_for(asyncio.gather(discard(), process.wait()), 1)
        finally:
            _WORKERS.release()


async def decode_chat_images(data, context):
    pending = collect_chat_images(data)  # All cheap limits before any worker.
    result, normalized_bytes = {}, 0
    for position, candidate in pending.items():
        result[position] = await decode_image(candidate, context)
        normalized_bytes += len(result[position].data)
        if normalized_bytes > MAX_NORMALIZED_BYTES:
            raise VisionError("unsupported_value", candidate.param)
    return MappingProxyType(result)


class _LimitedBuffer(io.BytesIO):
    def write(self, data):
        if self.tell() + len(data) > MAX_NORMALIZED_BYTES:
            raise VisionError("unsupported_value")
        return super().write(data)


def _decode_worker(data, mime):
    """Runs only under the subprocess resource limits, also callable by tests."""
    from PIL import Image, ImageFile, ImageOps, __version__
    import warnings
    if __version__ != PILLOW_VERSION:
        raise VisionError("model_unavailable", status=503)
    dimensions = _dimensions(data, mime)
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(data), formats=[FORMATS[mime]]) as check:
            if check.format != FORMATS[mime] or check.size != dimensions or getattr(check, "n_frames", 1) != 1:
                raise ValueError("image format, dimensions or animation")
            check.verify()
        with Image.open(io.BytesIO(data), formats=[FORMATS[mime]]) as image:
            image.load()
            orientation = image.getexif().get(274, 1)
            if type(orientation) is not int or orientation not in range(1, 9):
                raise ValueError("EXIF orientation")
            changed = orientation != 1 or mime == "image/webp"
            if orientation != 1:
                image = ImageOps.exif_transpose(image)
            if image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info:
                rgba = image.convert("RGBA")
                if rgba.getextrema()[3] != (255, 255):
                    white = Image.new("RGBA", image.size, (255, 255, 255, 255))
                    image = Image.alpha_composite(white, rgba).convert("RGB")
                    changed = True
            if changed:
                if image.mode not in ("1", "L", "LA", "P", "RGB", "RGBA", "I", "I;16"):
                    image = image.convert("RGB")
                buffer = _LimitedBuffer()
                image.save(buffer, format="PNG", compress_level=1, exif=b"")
                output, output_mime = buffer.getvalue(), "image/png"
            else:
                output, output_mime = data, mime
            return {"media_type": output_mime, "width": image.width, "height": image.height}, output


def _worker_main():
    resource.setrlimit(resource.RLIMIT_AS, (WORKER_MEMORY, WORKER_MEMORY))
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if len(sys.argv) != 3 or sys.argv[1] != "--decode" or sys.argv[2] not in FORMATS:
        raise ValueError("invalid decoder command")
    data = sys.stdin.buffer.read(MAX_SOURCE_BYTES + 1)
    if not data or len(data) > MAX_SOURCE_BYTES:
        raise ValueError("image size")
    metadata, output = _decode_worker(data, sys.argv[2])
    sys.stdout.buffer.write(json.dumps(metadata, separators=(",", ":")).encode() + b"\n")
    sys.stdout.buffer.write(output)


if __name__ == "__main__":
    try:
        _worker_main()
    except ImportError:
        raise SystemExit(3)
    except VisionError as error:
        raise SystemExit({"unsupported_value": 2, "model_unavailable": 3}.get(error.code, 1))
    except Exception:
        raise SystemExit(1)
