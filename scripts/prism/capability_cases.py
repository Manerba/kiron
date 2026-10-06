"""Closed, offline checks of the six measured Prism feature cases."""
import base64
import io
import importlib.util
from pathlib import Path


def require(value, message):
    if not value:
        raise ValueError(message)


def decode(data):
    from provider_transport import decode_provider_json
    return decode_provider_json(data)


def current_helper(name):
    if name not in {"probe-openai-features.py", "probe-openai-responses.py"}:
        raise ValueError("unknown measurement helper")
    spec = importlib.util.spec_from_file_location("capability_" + name.replace("-", "_"),
                                                  Path(__file__).with_name(name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native_response(raw, streaming):
    if not streaming:
        return decode(raw)
    frames = []
    lines = [line for line in raw.splitlines() if line and not line.startswith(b":")]
    require(lines and lines[-1] == b"data: [DONE]", "native stream has no final DONE")
    for line in lines[:-1]:
        require(line.startswith(b"data: ") and line != b"data: [DONE]", "invalid native frame order")
        frames.append(decode(line[6:]))
    result, message, calls = {}, {"role": "assistant", "content": "", "reasoning_content": ""}, {}
    finish = None
    for frame in frames:
        if frame.get("usage") is not None:
            require("usage" not in result, "duplicate native usage")
            result["usage"] = frame["usage"]
        if "__verbose" in frame:
            require("__verbose" not in result, "duplicate original token record")
            result["__verbose"] = frame["__verbose"]
        if "model" in frame:
            require(result.get("model", frame["model"]) == frame["model"], "native model changed")
            result["model"] = frame["model"]
        choices = frame.get("choices")
        require(type(choices) is list and len(choices) <= 1, "native choice shape")
        if not choices:
            continue
        choice = choices[0]
        require(type(choice["index"]) is int and choice["index"] == 0 and finish is None, "native data after finish")
        delta = choice.get("delta", {})
        for key in ("content", "reasoning_content"):
            part = delta.get(key)
            require(part is None or type(part) is str, "native text type")
            message[key] += part or ""
        for item in delta.get("tool_calls", []):
            index = item["index"]
            require(type(index) is int and 0 <= index < 2, "native tool index")
            call = calls.setdefault(index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
            for key in ("id", "type"):
                if key in item:
                    require(not call[key] or call[key] == item[key], "native tool identity changed")
                    call[key] = item[key]
            for key in ("name", "arguments"):
                call["function"][key] += item.get("function", {}).get(key, "")
        if choice.get("finish_reason") is not None:
            finish = choice["finish_reason"]
    require(finish in {"stop", "length", "tool_calls"}, "missing native finish")
    message["tool_calls"] = [calls[index] for index in sorted(calls)]
    result["choices"] = [{"index": 0, "message": message, "finish_reason": finish}]
    return result


def native_calls(directory, process_id, read_json, read_bytes, *, model_sha256):
    from kiron_common.local_inference import TokenUsage
    from prism_reasoning import ReasoningTokenRules, exact_reasoning_usage, exact_disabled_reasoning_usage
    features = current_helper("probe-openai-features.py")
    requests = sorted(directory.glob("native-*.request.json"))
    require(requests, "native trace missing")
    records = []
    for index, path in enumerate(requests, 1):
        require(path.name == f"native-{index:03d}.request.json", "native trace numbering differs")
        prefix = str(path).removesuffix(".request.json")
        request = read_json(path)
        status = read_json(Path(prefix + ".status.json"))["status"]
        raw = read_bytes(Path(prefix + ".response.raw"))
        require(read_bytes(Path(prefix + ".eof")) == b"", "native EOF proof missing")
        records.append((request, status, raw))
    calls, cursor, unauthorized = [], 0, 0
    while cursor < len(records):
        request, status, raw = records[cursor]
        require(request["method"] == "POST" and request["path"] == "/v1/chat/completions", "unexpected native call")
        body = request["body"]
        require(body["model"] == "kiron-prism-" + process_id, "native generation alias differs")
        if status == 401:
            unauthorized += 1
            require(unauthorized <= 2 and body["max_tokens"] == 1 and body["stream"] is False,
                    "unexpected unauthorized probe")
            cursor += 1
            continue
        require(status == 200 and unauthorized == 0 and cursor + 1 < len(records), "native completion failed")
        require(type(body.get("temperature")) is int and body["temperature"] == 0
                and type(body.get("stream")) is bool, "native sampling or stream control differs")
        value = native_response(raw, body["stream"])
        require(value["model"] == body["model"], "native result model differs")
        lookup, lookup_status, lookup_raw = records[cursor + 1]
        verbose, usage = value["__verbose"], value["usage"]
        tokens = verbose["tokens"]
        require(lookup["method"] == "POST" and lookup["path"] == "/tokenize" and lookup_status == 200
                and lookup["body"] == {"content": tokens, "with_pieces": True, "add_special": False, "parse_special": False},
                "original ID lookup request differs")
        pieces = decode(lookup_raw)["tokens"]
        require([piece["id"] for piece in pieces] == tokens, "original ID lookup response differs")
        native_usage = TokenUsage(usage["prompt_tokens"], usage["completion_tokens"],
                                  cached_input_tokens=usage["prompt_tokens_details"]["cached_tokens"])
        require(native_usage.total_tokens == usage["total_tokens"] and len(tokens) == native_usage.output_tokens,
                "native token arithmetic differs")
        enabled = body["chat_template_kwargs"]["enable_thinking"]
        require(type(enabled) is bool and body["return_tokens"] is True and body["verbose"] is True,
                "native exact usage controls missing")
        require(type(body["max_tokens"]) is int and 0 < native_usage.output_tokens <= body["max_tokens"] <= 128,
                "native budget differs")
        rules = ReasoningTokenRules(model_sha256, features.TEMPLATE_SHA,
            "<|im_start|>assistant\n<think>\n" if enabled else "<|im_start|>assistant\n<think>\n\n</think>\n\n",
            "prefilled" if enabled else "disabled", (248068, "<think>"), (248069, "</think>"),
            ((248046, "<|im_end|>"),), ((248058, "<tool_call>"), (248059, "</tool_call>")))
        message = value["choices"][0]["message"]
        if enabled:
            require(body["reasoning_budget_tokens"] == 32, "reasoning budget differs")
            exact = exact_reasoning_usage(verbose, native_usage, token_pieces=pieces,
                reasoning_text=message.get("reasoning_content", ""), final_text=message.get("content") or "",
                rules=rules, artifact_sha256=model_sha256, template_revision=features.TEMPLATE_SHA)
        else:
            require(body["reasoning_effort"] == "none", "disabled thinking control differs")
            exact = exact_disabled_reasoning_usage(verbose, native_usage, token_pieces=pieces, rules=rules,
                artifact_sha256=model_sha256, template_revision=features.TEMPLATE_SHA)
        calls.append({"body": body, "response": value, "tokens": tokens, "pieces": pieces, "usage": exact})
        cursor += 2
    require(unauthorized == 2, "missing foreign-generation probes")
    return calls


def public_chat(value, native):
    usage, exact = value["usage"], native["usage"]
    require((usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"],
             usage["prompt_tokens_details"]["cached_tokens"], usage["completion_tokens_details"]["reasoning_tokens"])
            == (exact.input_tokens, exact.output_tokens, exact.total_tokens, exact.cached_input_tokens, exact.reasoning_output_tokens),
            "public Chat usage differs from original token proof")
    require(len(value["choices"]) == 1, "public choice count")
    choice = value["choices"][0]
    require(choice["finish_reason"] == native["response"]["choices"][0]["finish_reason"], "public finish differs")
    message = choice["message"]
    require((message.get("content") or "") == (native["response"]["choices"][0]["message"].get("content") or ""),
            "public text differs from native")
    require("reasoning_content" not in message, "native private reasoning leaked")
    public_tools(message.get("tool_calls") or [], native)
    return message


def public_tools(calls, native):
    originals = native["response"]["choices"][0]["message"].get("tool_calls") or []
    require(len(calls) == len(originals), "public/native tool count differs")
    for call, original in zip(calls, originals):
        function = call["function"] if "function" in call else call
        identity = call["id"] if "function" in call else call["call_id"]
        wrapped = decode(original["function"]["arguments"].encode())
        require(set(wrapped) == {"payload"} and identity == original["id"]
                and function["name"] == original["function"]["name"]
                and decode(function["arguments"].encode()) == wrapped["payload"],
                "tool identity or unwrapped arguments differ")


def tool_cities(message):
    calls = message.get("tool_calls", [])
    require(calls and len({call["id"] for call in calls}) == len(calls), "tool IDs missing or duplicated")
    require(all(call["type"] == "function" and call["function"]["name"] == "weather" for call in calls), "wrong tool")
    return {call["id"]: decode(call["function"]["arguments"].encode())["city"] for call in calls}


def validate_cases(kind, results, calls):
    features = current_helper("probe-openai-features.py")
    responses = current_helper("probe-openai-responses.py")
    require(type(results) is dict, "feature results missing")
    budgets = [call["body"]["max_tokens"] for call in calls]
    measured = {"budgets": sorted(set(budgets))}
    if kind.startswith("public-responses"):
        names = (["budget-terminal"] if kind == "public-responses-budget"
                 else ["text", "text-stream", "tools-stream", "tools-replay", "tools-ordered-replay", "reasoning-stream"])
        require(len(calls) == len(names), "Responses native case count differs")
        keys = {"budget-events", "budget-terminal"} if kind == "public-responses-budget" else {
            "request-records", "tools-ordered-replay-scope", *names, *(name + "-request" for name in names),
            *(name + suffix for name in names if name.endswith("stream") for suffix in ("-events", "-terminal"))}
        require(set(results) == keys, "unexpected Responses case matrix")
        if kind == "public-responses-budget":
            terminal = responses.validate_budget_events(results["budget-events"])
            require(terminal == results["budget-terminal"], "budget terminal differs")
        for name, native in zip(names, calls):
            value = results[name]
            usage, exact = value["usage"], native["usage"]
            require((usage["input_tokens"], usage["output_tokens"], usage["total_tokens"],
                     usage["input_tokens_details"]["cached_tokens"], usage["output_tokens_details"]["reasoning_tokens"])
                    == (exact.input_tokens, exact.output_tokens, exact.total_tokens, exact.cached_input_tokens, exact.reasoning_output_tokens),
                    "Responses usage differs from original tokens")
            if name != "budget-terminal":
                require(value["status"] == "completed", "Responses did not complete")
                request = results[name + "-request"]
                require(request["max_output_tokens"] == native["body"]["max_tokens"] == responses.BUDGETS[name]
                        and request["temperature"] == native["body"]["temperature"] == 0,
                        "Responses request controls differ")
            output_text = "".join(part["text"] for item in value["output"] if item["type"] == "message"
                                  for part in item["content"] if part["type"] == "output_text")
            require(output_text == (native["response"]["choices"][0]["message"].get("content") or ""),
                    "Responses text differs from original native result")
            public_tools([item for item in value["output"] if item["type"] == "function_call"], native)
            if name.endswith("stream"):
                events = results[name + "-events"]
                require([item["sequence_number"] for item in events] == list(range(len(events)))
                        and events[-1]["type"] == "response.completed" and events[-1]["response"] == value
                        and sum(item["type"] in {"response.completed", "response.incomplete", "response.failed"} for item in events) == 1,
                        "Responses terminal sequence differs")
            if name in ("text", "text-stream", "tools-replay", "tools-ordered-replay"):
                text = "".join(part["text"] for item in value["output"] if item["type"] == "message"
                               for part in item["content"] if part["type"] == "output_text")
                require(("".join(text.split()).strip('`') if name.startswith("tools-") else text.strip())
                        == ("Berlin=ALPHA;Paris=BETA" if name.startswith("tools-") else "OK"),
                        "Responses final text differs")
            if name == "reasoning-stream":
                require("323" in output_text and exact.reasoning_output_tokens > 0 and any(item["type"] == "reasoning" for item in value["output"])
                        and all(not item["summary"] for item in value["output"] if item["type"] == "reasoning"),
                        "Responses reasoning trace missing")
        if kind == "public-responses":
            result = results["tools-stream"]
            by_id = {item["call_id"]: decode(item["arguments"].encode())["city"] for item in result["output"] if item["type"] == "function_call"}
            require(len(by_id) == 2 and set(by_id.values()) == {"Berlin", "Paris"}, "Responses parallel calls differ")
            replay = [item for item in results["tools-replay-request"]["input"] if item.get("type") == "function_call_output"]
            require([item["call_id"] for item in replay] == list(reversed(by_id))
                    and {by_id[item["call_id"]]: decode(item["output"].encode())["value"] for item in replay}
                    == {"Berlin": "ALPHA", "Paris": "BETA"}, "distinct reversed tool results missing")
            ordered = results["tools-ordered-replay-request"]
            generated_calls = [item for item in result["output"] if item["type"] == "function_call"]
            source_history = results["tools-stream-request"]["input"]
            expected_items = [*source_history, generated_calls[0], {'type': 'message', 'role': 'assistant',
                'content': [{'type': 'output_text', 'text': responses.ORDERED_HISTORY_TEXT, 'annotations': []}]},
                generated_calls[1], *replay, results["tools-replay-request"]["input"][-1]]
            require(ordered["input"] == expected_items and ordered["tool_choice"] == "none"
                    and ordered["parallel_tool_calls"] is False, "constructed mixed history input differs")
            require(results["tools-ordered-replay-scope"] == {'history': 'constructed',
                    'output_item_order': ['function_call', 'message', 'function_call'],
                    'calls_from': 'tools-stream', 'text': responses.ORDERED_HISTORY_TEXT},
                    "constructed history provenance missing")
            native = calls[names.index("tools-ordered-replay")]["body"]
            native_calls = [{"id": item["call_id"], "type": "function", "function": {
                "name": item["name"], "arguments": '{"payload":' + item["arguments"] + '}'}}
                for item in generated_calls]
            answers = {item["call_id"]: item["output"] for item in replay}
            expected_native = [*source_history,
                {"role": "assistant", "content": "", "tool_calls": [native_calls[0]]},
                {"role": "assistant", "content": responses.ORDERED_HISTORY_TEXT, "tool_calls": [native_calls[1]]},
                *[{"role": "tool", "content": answers[item["call_id"]], "tool_call_id": item["call_id"]}
                  for item in generated_calls], expected_items[-1]]
            require(native["messages"] == expected_native and native["stream"] is False and native["tool_choice"] == "none"
                    and native["parallel_tool_calls"] is False, "native mixed-turn segmentation or result binding differs")
            measured["constructed_history_order"] = ["function_call", "message", "function_call"]
        return measured

    names = {
        "public-tools": ["tool-named", "tool-single-stream", "tool-parallel", "tool-parallel-stream", "tool-roundtrip"],
        "public-vision": ["vision-two-images", "vision-two-images-stream"],
        "public-structured": ["json-object", "json-schema-stream"],
        "public-reasoning": ["reasoning", "reasoning-stream", "reasoning-length"],
    }.get(kind)
    require(names is not None and len(calls) == len(names), "feature case count differs")
    streams = {
        "public-tools": (False, True, False, True, False), "public-vision": (False, True),
        "public-structured": (False, True), "public-reasoning": (False, True, True),
    }[kind]
    keys = {"request-records", *names, *(name + "-events" for name, stream in zip(names, streams) if stream)}
    require(set(results) == keys and [call["body"]["stream"] for call in calls] == list(streams),
            "unexpected feature case matrix")
    for name, stream in zip(names, streams):
        if stream:
            events = results[name + "-events"]
            require(events and "chunk" in events and all(type(item) is str for item in events), "SDK stream events missing")
    records = results["request-records"]
    require(len(records) == len(calls) and len({record["id"] for record in records}) == len(records)
            and all(record["path"] == "/v1/chat/completions" and record["status"] == 200
                    and record["state"] == "completed" and record["error_code"] == ""
                    and record["tokens"] == call["usage"].output_tokens for record, call in zip(records, calls)),
            "public request completion records differ")
    messages = [public_chat(results[name], native) for name, native in zip(names, calls)]
    if kind == "public-tools":
        require([tool_cities(message) and sorted(tool_cities(message).values()) for message in messages[:4]]
                == [["Berlin"], ["Berlin"], ["Berlin", "Paris"], ["Berlin", "Paris"]], "named/parallel strict cases differ")
        require("OK" in (messages[-1]["content"] or ""), "tool roundtrip did not answer")
        require([call["body"]["parallel_tool_calls"] for call in calls[:4]] == [False, False, True, True],
                "parallel policy differs")
        for index, native in enumerate(calls[:4]):
            require(native["body"]["tool_choice"] == "required" and len(native["body"]["tools"]) == 1,
                    "named/required translation differs")
            schema = native["body"]["tools"][0]["function"]["parameters"]
            expected = {"type": "object", "properties": {"city": {"type": "string", "enum":
                        ["Berlin"] if index < 2 else ["Berlin", "Paris"]}}, "required": ["city"], "additionalProperties": False}
            require(schema == {"type": "object", "properties": {"payload": expected},
                               "required": ["payload"], "additionalProperties": False}, "strict wrapper schema differs")
        require(calls[-1]["body"]["tool_choice"] == "none", "tool none translation differs")
    elif kind == "public-vision":
        from PIL import Image
        totals = []
        for native, message, colors in zip(calls, messages, [("red", "blue"), ("blue", "red")]):
            content = native["body"]["messages"][0]["content"]
            require([part["type"] for part in content] == ["text", "image_url", "text", "image_url", "text"],
                    "native image order differs")
            images = [base64.b64decode(content[index]["image_url"]["url"].split(",", 1)[1], validate=True) for index in (1, 3)]
            for image in images:
                with Image.open(io.BytesIO(image)) as parsed:
                    require(parsed.size == (96, 96) and parsed.format in {"PNG", "JPEG"}, "native image shape differs")
                    parsed.load()
            totals.append(sum(map(len, images)))
            text = message["content"].lower()
            require(all(color in text for color in colors) and text.index(colors[0]) < text.index(colors[1]), "image answer order differs")
        measured["vision_bytes"] = max(totals)
    elif kind == "public-structured":
        from kiron_common.local_inference.json_schema import compile_schema, validate_instance
        from openai_generation import parse_output_format
        from prism_structured import native_output_format
        public_formats = ({"type": "json_object"}, {"type": "json_schema", "json_schema": {
            "name": "report", "schema": features.SCHEMA, "strict": True}})
        for native, public in zip(calls, public_formats):
            expected = native_output_format(parse_output_format(public))["response_format"]
            require(native["body"].get("response_format") == expected,
                    "native structured grammar or schema differs")
        require(type(decode(messages[0]["content"].encode())) is dict, "JSON object result differs")
        validate_instance(decode(messages[1]["content"].encode()), compile_schema(features.SCHEMA, strict=True))
    else:
        require(all("323" in message["content"] for message in messages[:2]), "reasoning answer missing")
        require(all(call["usage"].reasoning_output_tokens > 0 for call in calls), "exact reasoning count missing")
        require(calls[-1]["usage"].output_tokens == 4 and calls[-1]["usage"].reasoning_output_tokens == 4
                and results["reasoning-length"]["choices"][0]["finish_reason"] == "length", "reasoning length differs")
    return measured
