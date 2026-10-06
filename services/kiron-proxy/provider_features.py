"""Pure evidence checks shared by service and direct provider entry points."""
from kiron_common.local_inference import CapabilityName, ErrorCode, ImagePart, OutputFormatKind
from kiron_common.local_inference.json_schema import SchemaError, compile_schemas, schema_features
from provider_transport import failure


def validate_chat_request(request, capabilities, implementation, *, stream=False):
    """Direct adapter calls obey the same evidence/range boundary as the API."""
    deployment = request.model.deployment
    for name in (CapabilityName.CHAT, *((CapabilityName.STREAMING,) if stream else ())):
        if not capabilities.supports(name, deployment, implementation):
            raise failure(ErrorCode.UNSUPPORTED_CAPABILITY, "Text mapping is not verified", name.value)
    constraints = capabilities.by_name[CapabilityName.CHAT].constraints

    def check(name, value):
        constraint = constraints.get(name)
        if constraint is None:
            raise failure(ErrorCode.UNSUPPORTED_PARAMETER, "Parameter mapping is not verified", name)
        if not constraint.accepts(value):
            raise failure(ErrorCode.UNSUPPORTED_VALUE, "Parameter is outside the verified profile", name)

    check("max_output_tokens", request.options.max_output_tokens)
    profile = deployment.resource_profile
    if (request.options.max_output_tokens > 100000 or profile is not None
            and request.options.max_output_tokens > profile.context_tokens):
        raise failure(ErrorCode.UNSUPPORTED_VALUE, "Token budget exceeds the runtime profile", "max_output_tokens")
    for message in request.messages:
        check("roles", message.role.value)
    for name in ("temperature", "top_p", "frequency_penalty", "presence_penalty", "seed"):
        value = getattr(request.options.sampling, name)
        if value is not None:
            check(name, value)
    for value in request.options.sampling.stop:
        check("stop", value)


def _allowed(constraints, name, value, parameter, *, explicit=False, mismatch=ErrorCode.UNSUPPORTED_VALUE):
    constraint = constraints.get(name)
    if constraint is None or explicit and constraint.allowed_values is None:
        raise failure(ErrorCode.UNSUPPORTED_CAPABILITY, "Feature mapping is not verified", parameter)
    if not constraint.accepts(value):
        raise failure(mismatch, "Feature value is outside the verified profile", parameter)


def validate_features(request, capabilities, implementation):
    """Validate before health, discovery, load, admission, or provider I/O."""
    needed = set()
    tools = bool(request.tools or any(message.tool_calls or message.tool_call_id for message in request.messages))
    structured = request.options.output_format.kind is not OutputFormatKind.TEXT
    reasoning = request.options.reasoning
    active_reasoning = reasoning.effort not in (None, "none")
    if any(message.reasoning for message in request.messages) or reasoning.summary is not None:
        raise failure(ErrorCode.UNSUPPORTED_CAPABILITY, "Reasoning replay and summaries are not verified", "reasoning")
    if ((tools and structured) or active_reasoning and (tools or structured or request.options.sampling.stop)):
        raise failure(ErrorCode.UNSUPPORTED_CAPABILITY, "Requested feature combination is not verified", "reasoning" if active_reasoning else "response_format")
    if any(isinstance(part, ImagePart) for message in request.messages for part in message.content):
        needed.add(CapabilityName.VISION)
    if tools:
        needed.add(CapabilityName.FUNCTION_TOOLS)
    if request.options.parallel_tool_calls:
        needed.add(CapabilityName.PARALLEL_TOOLS)
    if structured:
        needed.add(CapabilityName.STRUCTURED_OUTPUT)
    if active_reasoning:
        needed.add(CapabilityName.REASONING)
    for name in needed:
        if not capabilities.supports(name, request.model.deployment, implementation):
            raise failure(ErrorCode.UNSUPPORTED_CAPABILITY, "Requested model capability is not verified", name.value)

    pairs = [(tool.parameters, tool.strict) for tool in request.tools]
    output = request.options.output_format
    if output.kind is OutputFormatKind.JSON_SCHEMA:
        pairs.append((output.schema, output.strict))
    try:
        compiled = compile_schemas(pairs)
    except SchemaError as exc:
        parameter = "tools" if exc.schema_index < len(request.tools) else "response_format"
        raise failure(exc.code, "Schema violates the supported contract", parameter) from None

    if tools:
        constraints = capabilities.by_name[CapabilityName.FUNCTION_TOOLS].constraints
        for name, value in (("tool_choice", request.tool_choice.kind.value), ("max_tools", len(request.tools))):
            _allowed(constraints, name, value, name)
        for index, tool in enumerate(request.tools):
            _allowed(constraints, "strict", tool.strict, f"tools[{index}].function.strict")
    if structured:
        constraints = capabilities.by_name[CapabilityName.STRUCTURED_OUTPUT].constraints
        _allowed(constraints, "formats", output.kind.value, "response_format.type", explicit=True)
        if output.kind is OutputFormatKind.JSON_SCHEMA:
            _allowed(constraints, "strict", output.strict, "response_format.json_schema.strict", explicit=True)
            for kind, values in schema_features(compiled[-1]).items():
                for value in values:
                    _allowed(constraints, "schema_" + kind, value, "response_format.json_schema.schema",
                             explicit=True, mismatch=ErrorCode.UNSUPPORTED_CAPABILITY)
    if active_reasoning:
        constraints = capabilities.by_name[CapabilityName.REASONING].constraints
        _allowed(constraints, "efforts", reasoning.effort, "reasoning_effort", explicit=True)
    elif reasoning.effort == "none":
        _allowed(capabilities.by_name[CapabilityName.CHAT].constraints,
                 "reasoning_effort", "none", "reasoning_effort", explicit=True)
