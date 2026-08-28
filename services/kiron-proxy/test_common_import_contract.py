from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent


def test_app_and_metrics_use_regular_fail_fast_common_imports() -> None:
    for name in ("app.py", "metrics.py"):
        source = (ROOT / name).read_text(encoding="utf-8")
        assert "from kiron_common." in source
        assert "_COMMON_SRC" not in source
        assert "sys.path" not in source
        assert "except ImportError" not in source


def test_proxy_consumes_service_inventory_without_direct_hf_cache_access() -> None:
    source = (ROOT / "app.py").read_text(encoding="utf-8")

    assert "service_huggingface_inventory(" in source
    assert "scan_huggingface_inventory" not in source
    assert "default_huggingface_hub_cache" not in source
