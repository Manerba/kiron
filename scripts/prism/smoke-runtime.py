#!/usr/bin/env python3
"""Bounded, direct Prism upstream probes; no KIron conformance or tool execution."""
import argparse
import base64
from copy import deepcopy
import http.client
import ipaddress
import json
import math
from pathlib import Path
import socket
import threading
import time
from urllib.parse import urlsplit

CASES = ("health", "text", "text_stream", "tool", "tool_stream", "parallel_tools", "parallel_tools_stream", "named_tool_choice", "strict_tool_enum",
         "tool_roundtrip", "json_object", "json_schema", "reasoning", "responses", "embeddings")
PRODUCTION_PORTS = {5001, 8505, 11434, 11435, 11436, 11437, 11440, 11441, 11442}
OUTPUT_LIMIT = 16 * 1024 * 1024
REQUEST_LIMIT = 4 * 1024 * 1024
MISSING_PROJECTOR = "image input is not supported - hint: if this is unexpected, you may need to provide the mmproj"
TOOL = {"type": "function", "function": {"name": "weather", "strict": True,
    "description": "Return the temperature for the requested city.", "parameters": {
        "type": "object", "properties": {"city": {"type": "string", "enum": ["Berlin", "Paris"]}},
        "required": ["city"], "additionalProperties": False}}}


class Gap(Exception):
    """An explicit upstream rejection or insufficient generation budget."""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def decode(raw):
    def invalid(value):
        raise ValueError(f"non-finite JSON value: {value}")
    return json.loads(raw, parse_constant=invalid)


def chat_usage(value):
    require(isinstance(value, dict), "missing Chat usage")
    require(all(type(value.get(key)) is int and value[key] >= 0
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")), "invalid usage counts")
    require(value["total_tokens"] == value["prompt_tokens"] + value["completion_tokens"], "inconsistent usage total")
    return value


def validate_base_url(value):
    url = urlsplit(value)
    require(url.scheme == "http" and url.hostname and url.port, "explicit http loopback IP and port required")
    require(ipaddress.ip_address(url.hostname).is_loopback, "non-loopback address refused")
    require(url.path in ("", "/") and not (url.query or url.fragment or url.username or url.password),
            "base URL must have no path, query, fragment, or credentials")
    require(1024 <= url.port <= 65535 and url.port not in PRODUCTION_PORTS, "reserved/production port refused")
    return url


def transport(base_url, path, payload, timeout, max_bytes):
    """Literal loopback HTTP: no environment proxy, DNS, redirect, or auth support."""
    url = validate_base_url(base_url)
    body = None if payload is None else json.dumps(payload, allow_nan=False).encode()
    require(body is None or len(body) <= REQUEST_LIMIT, "request byte limit exceeded")
    result = {"status": None, "headers": {}, "raw": b"", "error": None}
    start = time.monotonic()
    connection = http.client.HTTPConnection(url.hostname, url.port, timeout=timeout)
    timer, expired = None, threading.Event()
    try:
        connection.connect()
        sock = connection.sock

        def expire():
            expired.set()
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

        remaining = timeout - (time.monotonic() - start)
        if remaining <= 0:
            raise TimeoutError("total request timeout")
        timer = threading.Timer(remaining, expire)
        timer.daemon = True
        timer.start()
        connection.request("GET" if payload is None else "POST", path, body=body,
                           headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"})
        reply = connection.getresponse()
        result.update(status=reply.status, headers=dict(reply.getheaders()))
        while True:
            chunk = reply.read1(min(65536, max_bytes + 1 - len(result["raw"])))
            if not chunk:
                break
            result["raw"] += chunk
            require(len(result["raw"]) <= max_bytes, "response byte limit exceeded")
        if expired.is_set():
            raise TimeoutError("total request timeout")
    except (OSError, ValueError, http.client.HTTPException) as error:
        result["error"] = "total request timeout" if expired.is_set() else f"{type(error).__name__}: {error}"
    finally:
        if timer:
            timer.cancel()
        connection.close()
        result["elapsed_s"] = round(time.monotonic() - start, 6)
    return result


def stream_chat(raw):
    frames = raw.decode("utf-8").replace("\r\n", "\n").split("\n\n")
    calls, content, finish, done, identity, count, usage, started = {}, "", None, False, None, 0, None, False
    for frame in frames:
        lines = [line[5:].lstrip(" ") for line in frame.splitlines() if line.startswith("data:")]
        if not lines:
            continue
        require(not done, "data after [DONE]")
        data = "\n".join(lines)
        if data == "[DONE]":
            done = True
            continue
        chunk = decode(data)
        count += 1
        require("error" not in chunk and chunk.get("object") == "chat.completion.chunk", "invalid/error SSE chunk")
        require(isinstance(chunk.get("id"), str) and chunk["id"], "missing stream id")
        identity = identity or chunk["id"]
        require(identity == chunk["id"], "unstable stream id")
        if chunk.get("usage") is not None:
            usage = chat_usage(chunk["usage"])
        for choice in chunk["choices"]:
            require(choice.get("index") == 0 and finish is None, "invalid choice or data after finish")
            delta = choice["delta"]
            if "role" in delta:
                require(delta["role"] == "assistant", "invalid stream role")
                started = True
            content += delta.get("content") or ""
            for item in delta.get("tool_calls", []):
                index = item["index"]
                require(type(index) is int and 0 <= index < 64, "invalid tool index")
                call = calls.setdefault(index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                if item.get("id"):
                    require(not call["id"] or call["id"] == item["id"], "unstable tool id")
                    call["id"] = item["id"]
                require(item.get("type", "function") == "function", "invalid tool type")
                function = item.get("function", {})
                call["function"]["name"] += function.get("name", "")
                call["function"]["arguments"] += function.get("arguments", "")
                require(len(call["function"]["arguments"].encode()) <= 256 * 1024, "tool argument byte limit exceeded")
            finish = choice.get("finish_reason")
    require(started and done and finish is not None, "stream missing assistant role, finish_reason or [DONE]")
    require(sorted(calls) == list(range(len(calls))), "non-contiguous tool indices")
    return {"role": "assistant", "content": content, "tool_calls": [calls[key] for key in sorted(calls)]}, finish, count, chat_usage(usage)


def validate_tools(message, finish, parallel=False):
    require(finish == "tool_calls", "missing tool_calls finish_reason")
    calls = message.get("tool_calls", [])
    require(len(calls) == (2 if parallel else 1), "wrong tool call count")
    ids, cities = set(), set()
    for call in calls:
        require(isinstance(call.get("id"), str) and call["id"] and call["id"] not in ids, "missing/duplicate tool id")
        ids.add(call["id"])
        require(call.get("type") == "function" and call["function"]["name"] == "weather", "unexpected tool function")
        arguments = decode(call["function"]["arguments"])
        require(isinstance(arguments, dict) and set(arguments) == {"city"}, "invalid tool argument shape")
        require(arguments["city"] in ("Berlin", "Paris"), "invalid tool city")
        cities.add(arguments["city"])
    require(cities == ({"Berlin", "Paris"} if parallel else {"Berlin"}), "requested cities not returned")
    return calls


class Runner:
    def __init__(self, args, send=transport):
        validate_base_url(args.base_url)
        self.args, self.send, self.requests = args, send, []
        self.directory = Path(args.output_dir)
        self.directory.mkdir(parents=True, exist_ok=True)

    def ask(self, case, path, payload):
        prefix = f"{len(self.requests) + 1:02d}-{case}"
        request = {"url": self.args.base_url.rstrip("/") + path, "method": "GET" if payload is None else "POST", "json": payload}
        (self.directory / f"{prefix}.request.json").write_text(json.dumps(request, indent=2) + "\n")
        result = self.send(self.args.base_url, path, payload, self.args.timeout, self.args.max_output_bytes)
        raw = result.pop("raw")
        result.update(case=case, prefix=prefix, response_bytes=len(raw))
        self.requests.append(result)
        (self.directory / f"{prefix}.response.raw").write_bytes(raw)
        (self.directory / f"{prefix}.response.json").write_text(json.dumps(result, indent=2) + "\n")
        require(not result["error"], result["error"])
        status = result["status"]
        if case == "vision" and status == 500 and decode(raw) == {
                "error": {"code": 500, "type": "server_error", "message": MISSING_PROJECTOR}}:
            raise Gap("pinned upstream explicitly requires a vision projector")
        if status in (404, 405, 501):
            raise Gap(f"upstream endpoint unavailable: HTTP {status}")
        if status in (400, 422) and any(term in raw.lower() for term in (b"not supported", b"unsupported", b"not implemented")):
            raise Gap(f"upstream explicitly rejects capability: HTTP {status}")
        require(200 <= status < 300, f"upstream HTTP {status}")
        content_type = next((v for k, v in result["headers"].items() if k.lower() == "content-type"), "")
        is_stream = payload and payload.get("stream")
        require(("text/event-stream" if is_stream else "application/json") in content_type, "unexpected content type")
        return raw if is_stream else decode(raw)

    def probe(self, case):
        prompt = "Reply with the single word OK."
        body = {"model": self.args.model, "messages": [{"role": "user", "content": prompt}],
                "temperature": 0, "max_tokens": self.args.max_tokens, "reasoning_format": "deepseek",
                "reasoning_effort": self.args.reasoning_effort or "none"}
        if case == "health":
            require(self.ask(case, "/health", None).get("status") == "ok", "health not ok")
            return {}
        if case == "responses":
            reply = self.ask(case, "/v1/responses", {"model": self.args.model, "input": prompt,
                             "store": False, "max_output_tokens": self.args.max_tokens})
            require({"id", "created_at", "object", "model", "output", "parallel_tool_calls", "tool_choice", "tools", "usage"} <= reply.keys(), "missing required Responses fields")
            require(reply.get("object") == "response" and reply.get("status") == "completed", "incomplete/invalid Responses envelope")
            usage = reply["usage"]
            require(all(type(usage.get(k)) is int and usage[k] >= 0 for k in ("input_tokens", "output_tokens", "total_tokens")), "missing/invalid Responses token counts")
            require(usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"], "inconsistent Responses usage")
            for field, detail in (("input_tokens_details", "cached_tokens"), ("output_tokens_details", "reasoning_tokens")):
                require(isinstance(usage.get(field), dict) and type(usage[field].get(detail)) is int and usage[field][detail] >= 0, "missing/invalid Responses token details")
            require(reply["usage"].get("output_tokens", 0) < self.args.max_tokens, "Responses reached token limit; completed status not trusted")
            require(any(part.get("type") == "output_text" and part.get("text", "").strip()
                        for item in reply["output"] if item.get("type") == "message" for part in item["content"]), "missing Responses text")
            return {"response_id": reply.get("id")}
        if case == "embeddings":
            reply = self.ask(case, "/v1/embeddings", {"model": self.args.model, "input": ["smoke fixture"], "encoding_format": "float"})
            require(reply.get("object") == "list" and len(reply["data"]) == 1, "invalid embeddings envelope")
            vector = reply["data"][0]["embedding"]
            require(isinstance(vector, list) and vector and all(type(v) in (int, float) and math.isfinite(v) for v in vector), "invalid embedding vector")
            return {"dimensions": len(vector)}
        if case in ("tool", "tool_stream", "parallel_tools", "parallel_tools_stream", "tool_roundtrip", "named_tool_choice", "strict_tool_enum"):
            parallel = case in ("parallel_tools", "parallel_tools_stream")
            body.update(tools=[TOOL], tool_choice="required", parallel_tool_calls=parallel)
            body["messages"][0]["content"] = "Call weather for Berlin" + (" and Paris, once each, in parallel." if parallel else " now.")
            if case == "named_tool_choice":
                body["tools"] = [TOOL, {"type": "function", "function": {**TOOL["function"], "name": "echo", "description": "Repeat the city name."}}]
                body["tool_choice"] = {"type": "function", "function": {"name": "weather"}}
                body["messages"][0]["content"] = "Call echo for Berlin. Do not call weather."
            if case == "strict_tool_enum":
                body["tools"] = [deepcopy(TOOL)]
                body["tools"][0]["function"]["parameters"]["properties"]["city"]["enum"] = ["Berlin"]
                body["messages"][0]["content"] = "Call weather for Tokyo. Use Tokyo as the city argument, not Berlin."
        if case in ("json_object", "json_schema"):
            body["messages"][0]["content"] = 'Return JSON exactly {"ok":true}.'
            body["response_format"] = {"type": case}
            if case == "json_schema":
                body["response_format"]["json_schema"] = {"name": "smoke", "strict": True, "schema": {
                    "type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}}
        if case == "reasoning":
            body["messages"][0]["content"] = "Think briefly: what is 17 times 19? Give the result."
            body.update(chat_template_kwargs={"enable_thinking": True}, reasoning_budget_tokens=self.args.reasoning_budget_tokens,
                        return_tokens=True, verbose=True)
            require(self.args.reasoning_effort != "none", "reasoning probe conflicts with explicit reasoning_effort=none")
            if self.args.reasoning_effort is None:
                body.pop("reasoning_effort")
        if case == "vision":
            require(Path(self.args.vision_image).is_file(), "vision image must be a regular file")
            with Path(self.args.vision_image).open("rb") as image:
                raw = image.read(2 * 1024 * 1024 + 1)
            require(len(raw) <= 2 * 1024 * 1024, "vision image byte limit exceeded")
            mime = "image/png" if raw.startswith(b"\x89PNG\r\n\x1a\n") else "image/jpeg" if raw.startswith(b"\xff\xd8\xff") else None
            require(mime is not None, "vision fixture must be PNG or JPEG")
            prompt = ("Identify the image's dominant color. Reply with exactly one English color word and nothing else."
                      if self.args.vision_expect else "Describe the image briefly.")
            body["messages"][0]["content"] = [{"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(raw).decode()}"}}]
        if case.endswith("_stream"):
            body.update(stream=True, stream_options={"include_usage": True})
        reply = self.ask(case, "/v1/chat/completions", body)
        if body.get("stream"):
            message, finish, events, usage = stream_chat(reply)
        else:
            require(reply.get("object") == "chat.completion" and len(reply["choices"]) == 1, "invalid Chat envelope")
            message, finish, events = reply["choices"][0]["message"], reply["choices"][0]["finish_reason"], 0
            usage = chat_usage(reply.get("usage"))
        require(message.get("role") == "assistant", "invalid assistant role")
        if finish == "length":
            raise Gap("generation reached max_tokens; capability not established")
        if case in ("tool", "tool_stream", "parallel_tools", "parallel_tools_stream", "tool_roundtrip", "named_tool_choice", "strict_tool_enum"):
            calls = validate_tools(message, finish, case in ("parallel_tools", "parallel_tools_stream"))
            if case != "tool_roundtrip":
                return {"calls": len(calls), "stream_events": events, "usage": usage, "sample_only": True,
                        "constraint_enforcement": "not_established",
                        "source_gap": {"named_tool_choice": "pinned parser treats object tool_choice as auto",
                                       "strict_tool_enum": "pinned parser drops strict; string enums may be unconstrained"}.get(case)}
            body["messages"] += [message, {"role": "tool", "tool_call_id": calls[0]["id"], "content": '{"temperature_c":20}'}]
            body.update(tool_choice="none")  # A fixed fixture result; never dispatch a model-provided function.
            reply = self.ask(case, "/v1/chat/completions", body)
            message, finish = reply["choices"][0]["message"], reply["choices"][0]["finish_reason"]
            usage = chat_usage(reply.get("usage"))
            require("20" in (message.get("content") or ""), "roundtrip answer did not use fixture result")
        require(finish == "stop" and isinstance(message.get("content"), str) and message["content"].strip(), "missing completed text")
        if case in ("json_object", "json_schema"):
            value = decode(message["content"])
            require(value == {"ok": True} and type(value["ok"]) is bool, "JSON output did not match fixture/schema")
        observed = {"finish_reason": finish, "stream_events": events, "text_chars": len(message["content"]), "usage": usage}
        if case == "reasoning":
            require(any(isinstance(message.get(key), str) and message[key].strip() for key in ("reasoning_content", "reasoning")), "no separate native reasoning observed")
            native = reply["__verbose"]
            ids, raw_text = native["tokens"], native["content"]
            require(isinstance(ids, list) and ids and all(type(token) is int and token >= 0 for token in ids), "missing native output token IDs")
            require(isinstance(raw_text, str), "missing native raw text")
            observed["native_token_evidence"] = {"ids_in_raw_response": len(ids), "completion_tokens": usage["completion_tokens"],
                "id_count_matches_completion_tokens": len(ids) == usage["completion_tokens"], "tokens_predicted": native.get("tokens_predicted"),
                "raw_text_chars": len(raw_text), "think_open_char": raw_text.find("<think>"), "think_close_char": raw_text.find("</think>"),
                "reasoning_token_boundary": "not_established", "reasoning_tokens": None}
        if case == "vision":
            if not self.args.vision_expect:
                raise Gap("vision request completed; image accuracy requires --vision-expect fixture marker")
            require(self.args.vision_expect.strip().casefold() == message["content"].strip().casefold(),
                    "image answer must exactly match the expected color word")
        return observed

    def run(self, cases):
        results = []
        for case in cases:
            try:
                results.append({"case": case, "status": "pass", "observed": self.probe(case)})
            except (Gap, ValueError, KeyError, TypeError, OSError, IndexError, AttributeError) as error:
                results.append({"case": case, "status": "gap" if isinstance(error, Gap) else "fail", "detail": str(error)})
        summary = {"scope": "direct-upstream-smoke; pass validates a fixture observation, not a general capability guarantee or KIron conformance", "config": vars(self.args),
                   "results": results, "request_count": len(self.requests), "requests": self.requests}
        (self.directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:18089; production ports refused")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True, help="new empty evidence directory")
    parser.add_argument("--case", action="append", choices=(*CASES, "vision"))
    parser.add_argument("--vision-image")
    parser.add_argument("--vision-expect", help="expected complete English color word; absent means accuracy remains a gap")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-output-bytes", type=int, default=OUTPUT_LIMIT)
    parser.add_argument("--reasoning-effort", choices=("none", "minimal", "low", "medium", "high", "xhigh"))
    parser.add_argument("--reasoning-budget-tokens", type=int, default=32, help="pinned Prism-native reasoning probe control")
    args = parser.parse_args(argv)
    try:
        validate_base_url(args.base_url)
        require(0 < args.timeout <= 600 and 1 <= args.max_tokens <= 4096 and 1 <= args.max_output_bytes <= OUTPUT_LIMIT, "invalid bounded timeout/token/byte limit")
        require(1 <= args.reasoning_budget_tokens <= 4096, "invalid reasoning budget")
        require(not Path(args.output_dir).exists() or not any(Path(args.output_dir).iterdir()), "evidence directory must be empty")
        cases = args.case or list(CASES) + (["vision"] if args.vision_image else [])
        require("vision" not in cases or args.vision_image, "--vision-image required for vision")
        summary = Runner(args).run(cases)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({"results": summary["results"], "request_count": summary["request_count"]}, indent=2))
    return 1 if any(row["status"] != "pass" for row in summary["results"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
