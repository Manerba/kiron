"""Pure validation of the existing catalog Dense embedding profile."""
from kiron_common.local_inference import CapabilityName, ErrorCode
from kiron_common.local_inference.embedding import embedding_dimension
from provider_transport import failure


def validate_embedding_request(request, capabilities, implementation):
    try:
        dimension = embedding_dimension(request.model)
    except (TypeError, ValueError, KeyError):
        raise failure(ErrorCode.UNSUPPORTED_CAPABILITY, "A verified Dense embedding profile is required", "model") from None
    if not capabilities.supports(CapabilityName.EMBEDDINGS, request.model.deployment, implementation):
        raise failure(ErrorCode.UNSUPPORTED_CAPABILITY, "Embedding capability is not verified", "embeddings")
    constraints = capabilities.by_name[CapabilityName.EMBEDDINGS].constraints
    role = None if request.model.embedding_role is None else request.model.embedding_role.value
    selected_dimension = request.dimensions or dimension
    values = (("profiles", request.model.profile_id), ("roles", role), ("dimensions", selected_dimension),
              ("max_batch_size", len(request.inputs)))
    for name, value in values:
        constraint = constraints.get(name)
        if constraint is None or name in {"profiles", "roles", "dimensions"} and constraint.allowed_values is None:
            raise failure(ErrorCode.UNSUPPORTED_CAPABILITY, "Embedding mapping is not verified", name)
        if not constraint.accepts(value):
            raise failure(ErrorCode.UNSUPPORTED_VALUE, "Embedding value is outside the verified profile", name)
    if selected_dimension != dimension:
        reduction = constraints.get("native_dimensions")
        if reduction is None or reduction.allowed_values is None or not reduction.accepts(selected_dimension):
            raise failure(ErrorCode.UNSUPPORTED_CAPABILITY, "Native dimension reduction is not verified", "dimensions")
    limit = constraints.get("max_input_characters")
    if limit is None:
        raise failure(ErrorCode.UNSUPPORTED_CAPABILITY, "Embedding input limit is not verified", "input")
    if not 1 <= len(request.inputs) <= 512:
        raise failure(ErrorCode.INVALID_REQUEST, "Embedding batch exceeds the input limit", "input")
    for text in request.inputs:
        if not text or len(text) > 32768:
            raise failure(ErrorCode.INVALID_REQUEST, "Invalid embedding input length", "input")
        try:
            text.encode("utf-8")
        except UnicodeError:
            raise failure(ErrorCode.INVALID_REQUEST, "Invalid embedding input encoding", "input") from None
        if not limit.accepts(len(text)):
            raise failure(ErrorCode.UNSUPPORTED_VALUE, "Embedding input exceeds the verified limit", "input")
    return selected_dimension
