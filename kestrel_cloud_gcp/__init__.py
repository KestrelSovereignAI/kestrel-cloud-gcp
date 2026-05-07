"""Kestrel Cloud GCP — GCP Compute Engine provider for Kestrel Sovereign.

Extracted from kestrel-sovereign as a standalone cloud-provider package.
Registers ``GCPComputeFeature`` via the ``kestrel_sovereign.features``
entry-point group; auto-discovered when installed alongside
kestrel-sovereign.
"""

from importlib.metadata import PackageNotFoundError, version as _version

from .compute import (
    GCPComputeEngineManager,
    GCPComputeManager,
    GCPComputeManagerError,
    GCPComputeSession,
    GPUProfile,
    InstanceStatus,
)
from .compute.feature import GCPComputeFeature

try:
    __version__ = _version("kestrel-cloud-gcp")
except PackageNotFoundError:
    __version__ = "0.0.0+local"

__all__ = [
    "GCPComputeFeature",
    "GCPComputeEngineManager",
    "GCPComputeManager",
    "GCPComputeManagerError",
    "GCPComputeSession",
    "GPUProfile",
    "InstanceStatus",
    "__version__",
]
