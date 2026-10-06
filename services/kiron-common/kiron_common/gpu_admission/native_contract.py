"""Resource and completion contracts for KIron's native GPU services.

Memory budgets describe additional allocations, not already resident weights.
An HTTP error is an outcome; it proves termination only with the matching
service-issued operation receipt and a fully consumed, closed response.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import re


OPERATION_HEADER = "X-Kiron-Operation-Id"
OVERLAY_HEADER = "X-Kiron-Overlay-Token"
COMPLETION_HEADER = "X-Kiron-Operation-Terminated"
OPERATION_PATH = "/_internal/operations/"


def valid_operation_id(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{32}", value) is not None


@dataclass(frozen=True, slots=True)
class NativeMemoryBudget:
    load_bytes: int
    request_bytes: int
    headroom_bytes: int

    @classmethod
    def parse(cls, value: object) -> "NativeMemoryBudget":
        fields = {"load_bytes", "request_bytes", "headroom_bytes"}
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("gpu_memory must contain load_bytes, request_bytes and headroom_bytes")
        if any(type(value[key]) is not int or value[key] <= 0 for key in fields):
            raise ValueError("gpu_memory budgets must be positive integer bytes")
        return cls(**value)

    def additional_bytes(self, *, loading: bool) -> int:
        return self.request_bytes + (self.load_bytes if loading else 0)
