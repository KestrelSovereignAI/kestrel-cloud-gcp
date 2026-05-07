"""
GCP Compute Engine GPU Feature for Kestrel agents.

Exposes GCP Compute GPU orchestration via the tool system, providing commands
for starting, stopping, and managing GPU instances with persistent disk support.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from kestrel_sdk.features.base import Feature, tool
from kestrel_sdk.tools.result import ToolResult
from kestrel_cloud_gcp.compute.manager import GCPComputeEngineManager as GCPComputeManager
from kestrel_cloud_gcp.compute.models import (
    GCPComputeManagerError,
    InstanceStatus,
)
from kestrel_sdk.llm.types import BackendType
from kestrel_sdk.tools.base import ToolCategory

logger = logging.getLogger(__name__)


class GCPComputeFeature(Feature):
    """Feature layer exposing GCP Compute GPU orchestration via the tool system."""

    @property
    def tool_description(self) -> str:
        return (
            "Manage GCP Compute Engine GPU instances - start and stop on-demand GPU VMs, "
            "check status, manage persistent storage, and route LLM traffic. "
            "Supports spot instances for cost savings."
        )

    async def initialize(self):
        """Initialize the GCP Compute Engine manager."""
        self.manager = GCPComputeManager()
        self.llm_service = getattr(self.agent, "llm_service", None)
        if not self.llm_service:
            logger.warning("LLMService not available; GPU routing disabled")

    @tool(
        name="manage_gcp",
        description="Start, stop, or inspect GCP Compute GPU instances (usage: !gcp <action> [...]).",
        category=ToolCategory.SYSTEM,
        command_prefix="!gcp",
    )
    async def manage_gcp(
        self,
        action: str = "status",
        profile: str = "",
        model_name: str = "",
        ttl_seconds: str = "",
        use_spot: str = "true",
    ) -> ToolResult:
        """
        Main entry point for GCP Compute instance management.

        Actions:
            - status: Show current session status
            - on/start: Start a new GPU instance
            - off/stop: Stop and terminate current instance
            - list: List all Kestrel instances
            - disks: List persistent disks

        Examples:
            !gcp status
            !gcp on profile=training
            !gcp on profile=training use_spot=false
            !gcp off
            !gcp list
            !gcp disks
        """
        action_normalized = (action or "status").lower()

        if action_normalized in {"status"}:
            return await self._status()

        if action_normalized in {"on", "start"}:
            return await self._start(
                profile_name=profile,
                model_name=model_name,
                ttl_seconds=ttl_seconds,
                use_spot=use_spot.lower() in {"true", "yes", "1", ""},
            )

        if action_normalized in {"off", "stop"}:
            return await self._stop()

        if action_normalized in {"list", "instances"}:
            return await self._list_instances()

        if action_normalized in {"disks"}:
            return await self._list_disks()

        return ToolResult.failed(
            f"Unknown action: {action}",
            data={
                "available_actions": ["status", "on", "off", "list", "disks"],
            },
        )

    async def _status(self) -> ToolResult:
        """Get current session status."""
        try:
            status = await self.manager.get_status()

            # Add cost estimate if session is active
            if status.get("status") not in {"offline", "terminated"}:
                started_at = status.get("started_at")
                cost_per_hr = status.get("actual_cost_per_hr")
                if started_at and cost_per_hr:
                    elapsed_seconds = (
                        datetime.now(timezone.utc)
                        - datetime.fromisoformat(str(started_at))
                    ).total_seconds()
                    elapsed_hours = elapsed_seconds / 3600
                    status["estimated_cost"] = f"${elapsed_hours * cost_per_hr:.4f}"

            return ToolResult.ok(
                confirmation=f"GCP session status: {status.get('status', 'unknown')}",
                data=status,
            )

        except GCPComputeManagerError as e:
            return ToolResult.failed(str(e))

    async def _start(
        self,
        profile_name: str,
        model_name: str = "",
        ttl_seconds: str = "",
        use_spot: bool = True,
    ) -> ToolResult:
        """Start a new GPU instance."""
        if not profile_name:
            profiles = list(self.manager.profiles.keys())
            return ToolResult.failed(
                "Profile required",
                data={
                    "available_profiles": profiles,
                    "usage": "!gcp on profile=training",
                },
            )

        try:
            ttl = self._coerce_optional_int(ttl_seconds)

            result = await self.manager.start_session(
                task_profile=profile_name,
                model_name=model_name or None,
                ttl_seconds=ttl,
                use_spot=use_spot,
            )

            # Register with LLM router if available
            if self.llm_service and result.get("inference_url"):
                try:
                    self.llm_service.switch_backend(
                        BackendType.REMOTE_GPU,
                        config={
                            "base_url": result["inference_url"],
                            "model": result.get("model_name"),
                            "ttl_seconds": result.get("remaining_ttl_seconds"),
                            "metadata": {
                                "provider": "gcp",
                                "instance": result.get("instance_name"),
                            },
                        },
                    )
                    result["llm_routing"] = "enabled"
                except (KeyError, ValueError, TypeError) as e:
                    logger.warning(f"Invalid LLM routing parameters: {e}", exc_info=True)
                    result["llm_routing"] = f"failed: {e}"
                except Exception as e:
                    logger.warning(f"Failed to register with LLM router: {e}", exc_info=True)
                    result["llm_routing"] = f"failed: {e}"

            instance_name = result.get("instance_name", "")
            return ToolResult.ok(
                confirmation=(
                    f"Started GCP instance{f' {instance_name}' if instance_name else ''} "
                    f"(profile: {profile_name})"
                ),
                data=result,
            )

        except GCPComputeManagerError as e:
            return ToolResult.failed(str(e))

    async def _stop(self) -> ToolResult:
        """Stop the current instance."""
        try:
            # Unregister from LLM router
            if self.llm_service:
                try:
                    self.llm_service.switch_backend(BackendType.CLOUD)
                except (KeyError, ValueError, TypeError) as e:
                    logger.warning(f"Invalid LLM routing state: {e}", exc_info=True)
                except Exception as e:
                    logger.warning(f"Failed to unregister from LLM router: {e}", exc_info=True)

            result = await self.manager.stop_session()
            return ToolResult.ok(
                confirmation="Stopped GCP session",
                data=result,
            )

        except GCPComputeManagerError as e:
            return ToolResult.failed(str(e))

    async def _list_instances(self) -> ToolResult:
        """List all Kestrel instances in the project."""
        try:
            instances = await self.manager.list_instances()
            return ToolResult.ok(
                confirmation=f"Listed {len(instances)} GCP instance(s)",
                data={
                    "instances": instances,
                    "count": len(instances),
                },
            )
        except (KeyError, ValueError) as e:
            return ToolResult.failed(f"Invalid instance data: {e}")
        except Exception as e:
            logger.error(f"Unexpected error listing instances: {e}", exc_info=True)
            return ToolResult.failed(str(e))

    async def _list_disks(self) -> ToolResult:
        """List all Kestrel persistent disks."""
        try:
            disks = await self.manager.list_disks()
            return ToolResult.ok(
                confirmation=f"Listed {len(disks)} GCP persistent disk(s)",
                data={
                    "disks": disks,
                    "count": len(disks),
                },
            )
        except (KeyError, ValueError) as e:
            return ToolResult.failed(f"Invalid disk data: {e}")
        except Exception as e:
            logger.error(f"Unexpected error listing disks: {e}", exc_info=True)
            return ToolResult.failed(str(e))

    def _coerce_optional_int(self, value: str) -> Optional[int]:
        """Convert string to int, returning None if empty or invalid."""
        if not value or value.strip() == "":
            return None
        try:
            return int(value)
        except ValueError:
            return None
