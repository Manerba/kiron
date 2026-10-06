"""Real bounded decoder tests; no model, provider or network traffic."""
import asyncio
import base64
import hashlib
import io
from pathlib import Path
import struct
import subprocess
import sys
import threading
import time
import unittest
from unittest import mock

from PIL import Image, ImageOps
from kiron_common.local_inference import RequestContext
import openai_vision as vision


def picture(fmt="PNG", *, mode="RGB", size=(7, 5), color=(17, 34, 51), **options):
    image = Image.new(mode, size, color)
    output = io.BytesIO()
    image.save(output, format=fmt, **options)
    return output.getvalue()


def url(data, mime="image/png", **fields):
    return {"url": f"data:{mime};base64,{base64.b64encode(data).decode()}", **fields}


def request(*parts):
    return {"model": "fixture", "messages": [{"role": "user", "content": list(parts)}]}


def part(data, mime="image/png", **fields):
    return {"type": "image_url", "image_url": url(data, mime, **fields)}


def context(seconds=5):
    return RequestContext("vision-fixture", time.monotonic() + seconds, asyncio.Event())


class ImagePreflightTests(unittest.TestCase):
    def test_exact_data_format_and_canonical_base64_only(self):
        data = picture()
        for reference, code in (("http://127.0.0.1/a", "unsupported_capability"),
                ("https://example.invalid/a", "unsupported_capability"), ("file:///tmp/a", "unsupported_capability"),
                ("ftp://host/a", "unsupported_capability"), ("data:image/gif;base64,AAAA", "invalid_request"),
                ("data:image/svg+xml;base64,AAAA", "invalid_request"),
                (url(data)["url"] + "=", "invalid_request"), (url(data)["url"] + "\n", "invalid_request"),
                ("data:image/png;name=x;base64,AAAA", "invalid_request"),
                ("data:image/png;base64,", "invalid_request"), ("data:image/png;base64,ä", "invalid_request")):
            with self.subTest(reference=reference), self.assertRaises(vision.VisionError) as raised:
                vision.parse_image_url({"url": reference}, param="image", budget=vision.ImageBudget())
            self.assertEqual(raised.exception.code, code)
        for value in ({"url": url(data)["url"], "extra": None},):
            with self.assertRaises(vision.VisionError) as raised:
                vision.parse_image_url(value, param="image", budget=vision.ImageBudget())
            self.assertEqual(raised.exception.code, "unsupported_parameter")

    def test_mime_signature_and_detail_are_not_trusted(self):
        for value in (url(picture(), "image/jpeg"), {"url": None}, url(picture(), detail=True),
                      url(picture(), detail="invented"), url(b"not an image")):
            with self.subTest(value=value), self.assertRaises(vision.VisionError):
                vision.parse_image_url(value, param="image", budget=vision.ImageBudget())
        candidate = vision.parse_image_url(url(picture(), detail="high"), param="image", budget=vision.ImageBudget())
        self.assertEqual(candidate.detail, "high")  # Model evidence decides support later.

    def test_limits_apply_across_images_and_messages_before_workers(self):
        data = picture()
        self.assertEqual(len(vision.collect_chat_images(request(*(part(data) for _ in range(4))))), 4)
        with self.assertRaises(vision.VisionError):
            vision.collect_chat_images(request(*(part(data) for _ in range(5))))
        two = {"messages": [{"role": "user", "content": [part(data)]}, {"role": "user", "content": [part(data)]}]}
        with mock.patch.object(vision, "MAX_SOURCE_BYTES", len(data) * 2 - 1), self.assertRaises(vision.VisionError):
            vision.collect_chat_images(two)
        with mock.patch.object(vision, "MAX_TOTAL_PIXELS", 69), self.assertRaises(vision.VisionError):
            vision.collect_chat_images(two)

    def test_pixel_and_side_limits_are_checked_from_container_before_decode(self):
        original = picture()
        for width, height in ((8193, 1), (1, 8193), (4001, 4000), (0, 5)):
            forged = original[:16] + struct.pack(">II", width, height) + original[24:]
            with self.subTest(size=(width, height)), self.assertRaises(vision.VisionError):
                vision.collect_chat_images(request(part(forged)))
        # The dimensions boundary is independent from CRC/full-decode validation.
        exact = original[:16] + struct.pack(">II", 4000, 4000) + original[24:]
        self.assertEqual(vision._dimensions(exact, "image/png"), (4000, 4000))

    def test_closed_image_parts_and_user_only(self):
        p = part(picture())
        for role in ("assistant", "system", "developer", "tool"):
            with self.subTest(role=role), self.assertRaises(vision.VisionError):
                vision.collect_chat_images({"messages": [{"role": role, "content": [p]}]})
        with self.assertRaises(vision.VisionError) as raised:
            vision.collect_chat_images(request({**p, "extra": False}))
        self.assertEqual(raised.exception.code, "unsupported_parameter")

    def test_animated_png_webp_and_truncated_containers_rejected(self):
        for fmt, mime in (("PNG", "image/png"), ("WEBP", "image/webp")):
            output = io.BytesIO()
            Image.new("RGB", (3, 2), "red").save(output, format=fmt, save_all=True,
                append_images=[Image.new("RGB", (3, 2), "blue")], duration=20, loop=0)
            with self.subTest(format=fmt), self.assertRaises(vision.VisionError):
                vision.collect_chat_images(request(part(output.getvalue(), mime)))
        for fmt, mime in (("PNG", "image/png"), ("JPEG", "image/jpeg"), ("WEBP", "image/webp")):
            with self.subTest(format=fmt), self.assertRaises(vision.VisionError):
                vision.collect_chat_images(request(part(picture(fmt)[:-3], mime)))


class ImageDecodeTests(unittest.IsolatedAsyncioTestCase):
    async def test_opaque_unrotated_png_jpeg_remain_byte_identical(self):
        for fmt, mime in (("PNG", "image/png"), ("JPEG", "image/jpeg")):
            with self.subTest(format=fmt):
                raw = picture(fmt)
                images = await vision.decode_chat_images(request(part(raw, mime)), context())
                image = images[0, 0]
                self.assertEqual(image.data, raw)
                self.assertEqual(image.sha256, hashlib.sha256(raw).hexdigest())
                self.assertEqual((image.source_media_type, image.source_sha256), (mime, image.sha256))
                self.assertEqual((image.width, image.height, image.detail), (7, 5, "auto"))
                with self.assertRaises(TypeError):
                    images[0, 0] = image

    async def test_lossless_webp_becomes_png_with_same_pixels_and_source_identity(self):
        raw = picture("WEBP", lossless=True)
        image = (await vision.decode_chat_images(request(part(raw, "image/webp")), context()))[0, 0]
        self.assertEqual(image.media_type, "image/png")
        self.assertEqual(image.source_media_type, "image/webp")
        self.assertEqual(image.source_sha256, hashlib.sha256(raw).hexdigest())
        self.assertNotEqual(image.sha256, image.source_sha256)
        with Image.open(io.BytesIO(raw)) as before, Image.open(io.BytesIO(image.data)) as after:
            self.assertEqual(before.convert("RGB").tobytes(), after.convert("RGB").tobytes())

    async def test_all_exif_orientations_are_exact_transpositions_without_resizing(self):
        original = Image.new("RGB", (3, 2))
        original.putdata([(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)])
        for orientation in range(1, 9):
            with self.subTest(orientation=orientation):
                exif = Image.Exif()
                exif[274] = orientation
                output = io.BytesIO()
                original.save(output, "JPEG", quality=100, subsampling=0, exif=exif)
                raw = output.getvalue()
                image = (await vision.decode_chat_images(request(part(raw, "image/jpeg")), context()))[0, 0]
                with Image.open(io.BytesIO(raw)) as before, Image.open(io.BytesIO(image.data)) as after:
                    expected = ImageOps.exif_transpose(before)
                    self.assertEqual(after.size, expected.size)
                    self.assertEqual(after.convert("RGB").tobytes(), expected.convert("RGB").tobytes())
                    self.assertEqual(after.getexif().get(274, 1), 1)
                self.assertEqual(image.media_type, "image/jpeg" if orientation == 1 else "image/png")

    async def test_alpha_and_palette_transparency_are_composed_onto_white(self):
        for fmt, mime in (("PNG", "image/png"), ("WEBP", "image/webp")):
            rgba = Image.new("RGBA", (3, 1))
            rgba.putdata([(255, 0, 0, 0), (255, 0, 0, 128), (0, 0, 255, 255)])
            output = io.BytesIO()
            rgba.save(output, fmt, **({"lossless": True} if fmt == "WEBP" else {}))
            result = (await vision.decode_chat_images(request(part(output.getvalue(), mime)), context()))[0, 0]
            with Image.open(io.BytesIO(result.data)) as decoded:
                self.assertEqual(list(decoded.get_flattened_data()), [(255, 255, 255), (255, 127, 127), (0, 0, 255)])
        palette = Image.new("P", (1, 1), 0)
        palette.putpalette([255, 0, 0] + [0] * 765)
        output = io.BytesIO()
        palette.save(output, "PNG", transparency=0)
        result = (await vision.decode_chat_images(request(part(output.getvalue())), context()))[0, 0]
        with Image.open(io.BytesIO(result.data)) as decoded:
            self.assertEqual(decoded.getpixel((0, 0)), (255, 255, 255))

    async def test_real_full_decode_rejects_crc_corruption_after_valid_headers(self):
        raw = bytearray(picture())
        raw[29] ^= 0xFF  # Corrupt IHDR CRC without changing dimensions.
        self.assertEqual(len(vision.collect_chat_images(request(part(bytes(raw))))), 1)
        with self.assertRaises(vision.VisionError):
            await vision.decode_chat_images(request(part(bytes(raw))), context())

    async def test_multiple_parts_are_keyed_in_original_positions_across_messages(self):
        raw = picture()
        data = request({"type": "text", "text": "before"}, part(raw), {"type": "text", "text": "middle"}, part(raw))
        data["messages"].append({"role": "user", "content": [part(raw)]})
        images = await vision.decode_chat_images(data, context())
        self.assertEqual(list(images), [(0, 1), (0, 3), (1, 0)])

    async def test_decoded_parts_enter_canonical_chat_in_exact_original_order(self):
        from kiron_common.local_inference import ImagePart, TextPart
        from openai_wire import ApiError, parse_chat
        from prism_vision import message_content
        raw = picture("WEBP", lossless=True)
        data = request({"type": "text", "text": "a"}, {"type": "text", "text": "b"},
            part(raw, "image/webp"), {"type": "text", "text": "c"}, part(picture(), detail="low"))
        images = await vision.decode_chat_images(data, context())
        parsed = parse_chat(data, image_parts=images)
        content = parsed.messages[0].content
        self.assertEqual([type(value) for value in content], [TextPart, TextPart, ImagePart, TextPart, ImagePart])
        self.assertIs(content[2], images[0, 2])
        self.assertIs(content[4], images[0, 4])
        native = message_content(parsed.messages[0])
        self.assertEqual([value["type"] for value in native], ["text", "image_url", "text", "image_url"])
        self.assertEqual((native[0]["text"], native[2]["text"]), ("ab", "c"))
        self.assertEqual(native[3]["image_url"]["detail"], "low")
        for wrong in ({(0, 2): images[0, 2]}, {**images, (1, 0): images[0, 2]}):
            with self.assertRaises(ApiError):
                parse_chat(data, image_parts=wrong)

    async def test_invalid_preflight_and_capacity_do_not_spawn_worker(self):
        with mock.patch.object(vision.asyncio, "create_subprocess_exec", new_callable=mock.AsyncMock) as spawn:
            with self.assertRaises(vision.VisionError):
                await vision.decode_chat_images(request(part(picture(), "image/jpeg")), context())
            with mock.patch.object(vision, "_WORKERS", threading.BoundedSemaphore(0)), self.assertRaises(vision.VisionError) as raised:
                await vision.decode_chat_images(request(part(picture())), context())
            self.assertEqual((raised.exception.code, raised.exception.status), ("overloaded", 429))
            spawn.assert_not_called()

    async def test_normalized_byte_sum_has_explicit_bound(self):
        data = picture()
        pending = vision.collect_chat_images(request(part(data), part(data)))
        first = await vision.decode_image(pending[0, 0], context())
        with mock.patch.object(vision, "decode_image", new_callable=mock.AsyncMock, return_value=first), \
                mock.patch.object(vision, "MAX_NORMALIZED_BYTES", len(first.data) * 2 - 1), self.assertRaises(vision.VisionError) as raised:
            await vision.decode_chat_images(request(part(data), part(data)), context())
        self.assertEqual(raised.exception.code, "unsupported_value")

    async def test_decoder_output_is_bounded_while_reading_and_encoding(self):
        reader = asyncio.StreamReader()
        reader.feed_data(b"x" * 101)
        reader.feed_eof()
        with self.assertRaises(vision.VisionError) as raised:
            await vision._read_bounded(reader, 100)
        self.assertEqual(raised.exception.code, "unsupported_value")
        with mock.patch.object(vision, "MAX_NORMALIZED_BYTES", 8), self.assertRaises(vision.VisionError) as raised:
            vision._decode_worker(picture("WEBP", lossless=True), "image/webp")
        self.assertEqual(raised.exception.code, "unsupported_value")

    async def test_worker_dependency_and_expansion_errors_are_not_generic_bad_images(self):
        real_spawn = asyncio.create_subprocess_exec
        for exit_code, code, status in ((2, "unsupported_value", 400), (3, "model_unavailable", 503)):
            async def reject(*args, **kwargs):
                return await real_spawn(sys.executable, "-I", "-c", f"import sys;sys.stdin.buffer.read();sys.exit({exit_code})", **kwargs)
            with self.subTest(code=code), mock.patch.object(vision.asyncio, "create_subprocess_exec", side_effect=reject), \
                    self.assertRaises(vision.VisionError) as raised:
                await vision.decode_chat_images(request(part(picture())), context())
            self.assertEqual((raised.exception.code, raised.exception.status), (code, status))

    async def test_excess_worker_output_is_discarded_and_process_reaped(self):
        real_spawn, processes = asyncio.create_subprocess_exec, []
        async def excessive(*args, **kwargs):
            process = await real_spawn(sys.executable, "-I", "-c",
                'import sys,time;sys.stdin.buffer.read();sys.stdout.buffer.write(b"x"*10_000_000);'
                'sys.stdout.buffer.flush();time.sleep(30)', **kwargs)
            processes.append(process)
            return process
        started = time.monotonic()
        with mock.patch.object(vision.asyncio, "create_subprocess_exec", side_effect=excessive), \
                mock.patch.object(vision, "MAX_NORMALIZED_BYTES", 100), self.assertRaises(vision.VisionError) as raised:
            await vision.decode_chat_images(request(part(picture())), context())
        self.assertEqual(raised.exception.code, "unsupported_value")
        self.assertLess(time.monotonic() - started, 1.5)
        self.assertIsNotNone(processes[0].returncode)
        self.assertFalse(Path(f"/proc/{processes[0].pid}").exists())

    async def test_wall_timeout_cancellation_and_task_cancel_kill_and_reap_worker(self):
        real_spawn = asyncio.create_subprocess_exec
        for mode in ("deadline", "signal", "task"):
            processes, ready = [], asyncio.Event()
            async def blocked_worker(*args, **kwargs):
                process = await real_spawn(sys.executable, "-I", "-c", "import time; time.sleep(30)", **kwargs)
                processes.append(process)
                ready.set()
                return process
            ctx = context()
            started = time.monotonic()
            with self.subTest(mode=mode), mock.patch.object(vision.asyncio, "create_subprocess_exec", side_effect=blocked_worker), \
                    mock.patch.object(vision, "DECODE_SECONDS", .15):
                pending = asyncio.create_task(vision.decode_chat_images(request(part(picture())), ctx))
                await ready.wait()
                if mode == "signal":
                    ctx.cancellation.set()
                elif mode == "task":
                    pending.cancel()
                with self.assertRaises(asyncio.CancelledError if mode == "task" else vision.VisionError):
                    await pending
                self.assertLess(time.monotonic() - started, 1.5)
                self.assertIsNotNone(processes[0].returncode)
                self.assertFalse(Path(f"/proc/{processes[0].pid}").exists())

    async def test_two_active_workers_exhaust_capacity_and_cancellation_restores_it(self):
        real_spawn = asyncio.create_subprocess_exec
        processes, ready = [], asyncio.Event()
        async def blocked_worker(*args, **kwargs):
            process = await real_spawn(sys.executable, "-I", "-c", "import time; time.sleep(30)", **kwargs)
            processes.append(process)
            if len(processes) == 2:
                ready.set()
            return process
        with mock.patch.object(vision.asyncio, "create_subprocess_exec", side_effect=blocked_worker):
            tasks = [asyncio.create_task(vision.decode_chat_images(request(part(picture())), context())) for _ in range(2)]
            try:
                await asyncio.wait_for(ready.wait(), 1)
                with self.assertRaises(vision.VisionError) as raised:
                    await vision.decode_chat_images(request(part(picture())), context())
                self.assertEqual((raised.exception.code, raised.exception.status), ("overloaded", 429))
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        self.assertTrue(all(process.returncode is not None for process in processes))
        self.assertEqual(len(await vision.decode_chat_images(request(part(picture())), context())), 1)

    def test_worker_entrypoint_enforces_actual_kernel_memory_and_cpu_limits(self):
        program = '''
import importlib.util, resource, sys
spec = importlib.util.spec_from_file_location('decoder', sys.argv[1])
decoder = importlib.util.module_from_spec(spec)
sys.modules['decoder'] = decoder
spec.loader.exec_module(decoder)
def check(data, mime):
    assert resource.getrlimit(resource.RLIMIT_AS) == (decoder.WORKER_MEMORY,) * 2
    assert resource.getrlimit(resource.RLIMIT_CPU) == (2, 2)
    assert resource.getrlimit(resource.RLIMIT_CORE) == (0, 0)
    try:
        bytearray(decoder.WORKER_MEMORY)
    except MemoryError:
        pass
    else:
        raise AssertionError('kernel did not enforce memory limit')
    return {'media_type': mime, 'width': 7, 'height': 5}, data
decoder._decode_worker = check
sys.argv = ['decoder', '--decode', 'image/png']
decoder._worker_main()
'''
        completed = subprocess.run([sys.executable, "-I", "-c", program, vision.__file__],
            input=picture(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3,
            env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"})
        self.assertEqual(completed.returncode, 0, completed.stderr.decode())
        self.assertEqual(completed.stdout.partition(b"\n")[2], picture())


if __name__ == "__main__":
    unittest.main()
