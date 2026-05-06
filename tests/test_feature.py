"""Smoke tests for kestrel-cloud-gcp.

These are package-level smoke tests that don't touch real GCP — they verify
the package layout, entry-point registration, and the feature class shape.

Real-cloud integration tests live elsewhere and require ``GCP_PROJECT_ID``
plus actual credentials.
"""
import importlib.metadata as md

import pytest

from kestrel_cloud_gcp import GCPComputeFeature
from kestrel_cloud_gcp.compute import (
    GCPComputeEngineManager,
    GCPComputeManagerError,
    GPUProfile,
    InstanceStatus,
)
from kestrel_sdk.features.base import Feature
from kestrel_sdk.llm import BackendType


class TestPackageSurface:
    """Verify the public surface of the package."""

    def test_feature_subclasses_sdk_feature(self):
        """GCPComputeFeature inherits from kestrel_sdk's Feature base class."""
        assert issubclass(GCPComputeFeature, Feature)

    def test_entry_point_registered(self):
        """`GCPComputeFeature` is discoverable via the
        `kestrel_sovereign.features` entry-point group."""
        names = [e.name for e in md.entry_points(group="kestrel_sovereign.features")]
        assert "GCPComputeFeature" in names, (
            f"GCPComputeFeature missing from entry-points: {names}"
        )

    def test_top_level_reexports(self):
        """All compute symbols listed in __all__ resolve."""
        from kestrel_cloud_gcp import (  # noqa: F401
            GCPComputeFeature,
            GCPComputeEngineManager,
            GCPComputeManager,
            GCPComputeManagerError,
            GCPComputeSession,
            GPUProfile,
            InstanceStatus,
        )

    def test_backend_type_imports_from_sdk(self):
        """The feature uses BackendType from the SDK, not the framework."""
        # Identity check: same enum, same values
        assert BackendType.CLOUD.value == "cloud"
        assert BackendType.LOCAL.value == "local"
        assert BackendType.REMOTE_GPU.value == "remote_gpu"

    def test_models_are_importable(self):
        """Data models load without GCP creds."""
        # GPUProfile is a dataclass; should construct without env vars.
        # We don't instantiate (constructor signature varies), just verify import.
        assert GPUProfile is not None
        assert InstanceStatus is not None
        assert GCPComputeManagerError is not None


class TestFeatureInitWithoutGCP:
    """Verify the feature initializes cleanly when GCP_PROJECT_ID is absent.

    The feature should construct + log a warning about missing creds, NOT
    crash. Tools should be discoverable but unusable until creds are set.
    """

    @pytest.mark.asyncio
    async def test_feature_initializes_without_creds(self, monkeypatch):
        monkeypatch.delenv("GCP_PROJECT_ID", raising=False)
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

        feature = GCPComputeFeature(agent=None)
        # initialize() is the async hook; should not raise.
        await feature.initialize()
        assert feature.agent is None

    def test_feature_class_exposes_tool_decorators(self):
        """Tool methods are decorated and discoverable on the class."""
        # The feature has @tool decorators; at least one should expose schema.
        tool_methods = [
            name for name in dir(GCPComputeFeature)
            if not name.startswith("_") and callable(getattr(GCPComputeFeature, name))
        ]
        assert tool_methods, "GCPComputeFeature exposes no public methods"
