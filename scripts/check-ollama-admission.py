#!/usr/bin/env python3
"""Read-only deploy preflight for the current admission contract."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services/kiron-common"))
from kiron_common.gpu_admission import AdmissionError, AdmissionStore


def main():
    store = AdmissionStore()
    path = store.root / "admission.json"
    if not path.exists() and not path.is_symlink():
        return 0
    try:
        store.snapshot()
    except AdmissionError as exc:
        print(f"FEHLER: Admission-Vertrag vor Deploy ungültig: {exc}. "
              "Betriebsablauf in docs/ollama-lifecycle-recovery.md beachten; "
              "keine laufenden Reservierungen löschen.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
