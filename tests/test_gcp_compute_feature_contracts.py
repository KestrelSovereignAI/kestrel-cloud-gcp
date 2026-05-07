"""Direct contracts for the GCP compute feature surface.

Every public ``@tool`` method (and the private dispatch helpers it
delegates to) returns a ``kestrel_sdk.tools.result.ToolResult``;
these tests pin both the success and failure shapes so the
framework's narration-honesty audit hook (#1042 layer 3) can trust
the wire format coming back."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_cloud_gcp.compute.feature import GCPComputeFeature
from kestrel_cloud_gcp.compute.models import GCPComputeManagerError
from kestrel_sdk.llm.types import BackendType
from kestrel_sdk.tools.result import ToolResult, ToolResultStatus


def _make_feature():
    feature = GCPComputeFeature(agent=SimpleNamespace())
    feature.manager = SimpleNamespace(
        profiles={"training": object(), "budget": object()},
        start_session=AsyncMock(),
        stop_session=AsyncMock(),
        get_status=AsyncMock(),
        list_instances=AsyncMock(),
        list_disks=AsyncMock(),
    )
    feature.llm_service = MagicMock()
    return feature


@pytest.mark.asyncio
async def test_start_requires_profile():
    feature = _make_feature()

    result = await feature._start(profile_name="")

    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.ERROR
    assert result.error == "Profile required"
    assert result.data["available_profiles"] == ["training", "budget"]


@pytest.mark.asyncio
async def test_start_forwards_model_ttl_and_enables_routing():
    feature = _make_feature()
    feature.manager.start_session.return_value = {
        "instance_name": "gcp-1",
        "inference_url": "http://gpu.example/v1",
        "model_name": "phi4",
        "remaining_ttl_seconds": 900,
    }

    result = await feature._start(
        profile_name="training",
        model_name="phi4",
        ttl_seconds="900",
        use_spot=False,
    )

    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.OK
    assert result.data["llm_routing"] == "enabled"
    feature.manager.start_session.assert_awaited_once_with(
        task_profile="training",
        model_name="phi4",
        ttl_seconds=900,
        use_spot=False,
    )
    feature.llm_service.switch_backend.assert_called_once_with(
        BackendType.REMOTE_GPU,
        config={
            "base_url": "http://gpu.example/v1",
            "model": "phi4",
            "ttl_seconds": 900,
            "metadata": {"provider": "gcp", "instance": "gcp-1"},
        },
    )


@pytest.mark.asyncio
async def test_start_skips_routing_without_inference_url():
    feature = _make_feature()
    feature.manager.start_session.return_value = {"instance_name": "gcp-2"}

    result = await feature._start(profile_name="training")

    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.OK
    assert "llm_routing" not in result.data
    feature.llm_service.switch_backend.assert_not_called()


@pytest.mark.asyncio
async def test_start_returns_manager_error_cleanly():
    feature = _make_feature()
    feature.manager.start_session.side_effect = GCPComputeManagerError("boom")

    result = await feature._start(profile_name="training")

    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.ERROR
    assert result.error == "boom"


@pytest.mark.asyncio
async def test_status_adds_estimated_cost_for_active_session():
    feature = _make_feature()
    started_at = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
    feature.manager.get_status.return_value = {
        "status": "running",
        "started_at": started_at,
        "actual_cost_per_hr": 2.0,
    }

    result = await feature._status()

    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.OK
    assert result.data["status"] == "running"
    assert result.data["estimated_cost"].startswith("$")


@pytest.mark.asyncio
async def test_stop_clears_remote_backend_before_stopping():
    feature = _make_feature()
    feature.manager.stop_session.return_value = {"status": "terminated"}

    result = await feature._stop()

    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.OK
    assert result.data == {"status": "terminated"}
    feature.llm_service.switch_backend.assert_called_once_with(BackendType.CLOUD)


@pytest.mark.asyncio
async def test_manage_gcp_reports_unknown_action():
    feature = _make_feature()

    result = await feature.manage_gcp(action="dance")

    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.ERROR
    assert result.data["available_actions"] == ["status", "on", "off", "list", "disks"]
