"""Cross-process GPU admission in the existing KIron runtime marker directory.

Reservations never expire into permission: an abandoned operation remains unknown
until its owner confirms backend termination. No process, network or GPU access
occurs on import. Callers supply a fresh resource measurement inside the lock.
"""

from .store import (
    AdmissionError, AdmissionStore, MemorySnapshot, RuntimeSecurity, Ticket,
    runtime_lock,
)

__all__ = [
    "AdmissionError", "AdmissionStore", "MemorySnapshot", "RuntimeSecurity",
    "Ticket", "runtime_lock",
]
