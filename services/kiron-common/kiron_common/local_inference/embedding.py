"""The existing Dense profile's role and dimension contract, without routing."""
from kiron_common.model_catalog import ModelEndpoint, ModelTask
from .identity import EmbeddingRole


def neutral_profile(metadata):
    contract = metadata.get("input_type", {})
    return (metadata.get("kind") == "dense"
            and metadata.get("verification", {}).get("status") == "verified"
            and contract.get("role_sensitive") is False
            and contract.get("required") is False
            and contract.get("missing_role_behavior") == "no_op")


def embedding_dimension(model):
    metadata = model.profile_metadata
    if (model.profile_id is None or model.task is not ModelTask.EMBEDDING
            or model.endpoint is not ModelEndpoint.EMBED or metadata.get("kind") != "dense"
            or metadata.get("verification", {}).get("status") != "verified"):
        raise ValueError("embedding requires a verified Dense profile")
    contract = metadata.get("input_type", {})
    if model.embedding_role is None:
        if not neutral_profile(metadata):
            raise ValueError("embedding requires an explicit profile role")
    elif not isinstance(model.embedding_role, EmbeddingRole) or model.embedding_role.value not in contract.get("supported", ()):
        raise ValueError("embedding role is outside the selected profile")
    dimensions = metadata.get("dimensions")
    if type(dimensions) is not int or not 1 <= dimensions <= 65536:
        raise ValueError("embedding profile has no exact dimension")
    return dimensions
