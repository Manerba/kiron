"""Measured service memory, kept separate from individual model allocations."""

from __future__ import annotations

import math


SERVICE_BACKENDS = {
    "kiron-embeddings": "kiron_embeddings",
    "kiron-deberta": "kiron_deberta",
}


def _nonnegative_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def build_service_memory(gpu_processes: object, *, embedding: object, deberta: object) -> dict:
    """Project one metrics snapshot without assigning shared memory to a model.

    Only exact systemd ownership qualifies; the graph's heuristic process labels
    are insufficient. Missing/partial measurements stay unknown independently
    for VRAM and RSS. A model switch has no stable set of resident models yet.
    """
    if not isinstance(gpu_processes, dict) or gpu_processes.get("state") != "ok":
        return {}
    processes = gpu_processes.get("data")
    if not isinstance(processes, list) or any(not isinstance(p, dict) for p in processes):
        return {}
    result = {}
    for service, health in (("kiron-embeddings", embedding), ("kiron-deberta", deberta)):
        if (not isinstance(health, dict) or health.get("running") is not True
                or health.get("status") != "ok" or health.get("loading_model") is not None):
            continue
        loaded = health.get("loaded_models")
        if (not isinstance(loaded, list) or not loaded
                or any(type(name) is not str or not name for name in loaded)
                or len(set(loaded)) != len(loaded)):
            continue
        owned = [p for p in processes if p.get("service") == service]
        if not owned:
            continue
        pids = [p.get("pid") for p in owned]
        if (any(type(pid) is not int or pid <= 0 for pid in pids)
                or len(set(pids)) != len(pids)):
            continue
        vram_known = all(p.get("vram_state") == "known" and _nonnegative_number(p.get("vram_mb"))
                         for p in owned)
        ram_known = all(type(p.get("rss_bytes")) is int and p["rss_bytes"] >= 0 for p in owned)
        result[SERVICE_BACKENDS[service]] = {
            "vram_gb": round(sum(p["vram_mb"] for p in owned) / 1024, 2) if vram_known else None,
            "ram_gb": round(sum(p["rss_bytes"] for p in owned) / (1024 ** 3), 2) if ram_known else None,
            "loaded_models": sorted(loaded),
            "process_count": len(owned),
        }
    return result
