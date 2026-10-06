"""Observe actual forward inputs in the exclusive serial model worker."""
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass

import torch


class TokenUsageError(ValueError):
    pass


@dataclass
class ForwardUsage:
    rows: int = 0
    tokens: int = 0

    def observe(self, features):
        if not isinstance(features, Mapping):
            raise TokenUsageError("forward inputs do not expose token tensors")
        ids, mask = features.get("input_ids"), features.get("attention_mask")
        integers = (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8)
        if (not isinstance(ids, torch.Tensor) or not isinstance(mask, torch.Tensor)
                or ids.ndim != 2 or mask.shape != ids.shape or ids.dtype not in integers
                or mask.dtype not in (*integers, torch.bool) or not ids.shape[0] or not ids.shape[1]
                or not bool(((mask == 0) | (mask == 1)).all())):
            raise TokenUsageError("forward token evidence is malformed")
        counts = mask.sum(dim=1)
        if not bool((counts > 0).all()):
            raise TokenUsageError("forward token evidence contains an empty input")
        self.rows += ids.shape[0]
        self.tokens += int(counts.sum().item())

    def finish(self, expected_rows):
        if self.rows != expected_rows or self.tokens < expected_rows:
            raise TokenUsageError("not every encoded input has exact forward token evidence")
        return self.tokens


@contextmanager
def capture_forward_usage(model):
    """No tokenization replay: count the precise padded tensors used by encode."""
    forward = getattr(model, "forward", None)
    if not callable(forward):
        raise TokenUsageError("encoder has no inspectable forward input")
    absent = object()
    previous = model.__dict__.get("forward", absent)
    usage = ForwardUsage()
    def observed(features, *args, **kwargs):
        usage.observe(features)
        return forward(features, *args, **kwargs)
    model.forward = observed
    try:
        yield usage
    finally:
        if previous is absent:
            del model.forward
        else:
            model.forward = previous
