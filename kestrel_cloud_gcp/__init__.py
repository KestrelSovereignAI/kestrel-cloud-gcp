"""Kestrel Cloud GCP — GCP Compute Engine provider for Kestrel Sovereign.

Extracted from kestrel-sovereign as a standalone cloud-provider package.
Registers ``GCPComputeFeature`` via the ``kestrel_sovereign.features``
entry-point group; auto-discovered when installed alongside
kestrel-sovereign.
"""

from .compute import (
    GCPComputeEngineManager,
    GCPComputeManager,
    GCPComputeManagerError,
    GCPComputeSession,
    GPUProfile,
    InstanceStatus,
)
from .compute.feature import GCPComputeFeature

__all__ = [
    "GCPComputeFeature",
    "GCPComputeEngineManager",
    "GCPComputeManager",
    "GCPComputeManagerError",
    "GCPComputeSession",
    "GPUProfile",
    "InstanceStatus",
]
