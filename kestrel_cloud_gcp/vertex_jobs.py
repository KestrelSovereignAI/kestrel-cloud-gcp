"""Content-free evidence for an exact Vertex v1 Custom Job readback."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

VERTEX_OPERATION_LABEL = "kestrel_operation_sha256"
_NUMBER = re.compile(r"[1-9][0-9]{0,19}")
_US_REGION = re.compile(r"us-[a-z]+[0-9]+")
_SHA256 = re.compile(r"[a-f0-9]{64}")
_RFC3339 = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.(?P<fraction>[0-9]{1,9}))?(?:Z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])"
)


class VertexJobObservationError(ValueError):
    """The provider observation cannot establish the original job's state."""


@dataclass(frozen=True, slots=True)
class VertexCustomJobLocator:
    """Exact resource name bound to a configured Google project number."""

    project_number: str
    region: str
    job_id: str

    def __post_init__(self) -> None:
        for value, pattern in (
            (self.project_number, _NUMBER),
            (self.region, _US_REGION),
            (self.job_id, _NUMBER),
        ):
            if not isinstance(value, str) or not pattern.fullmatch(value):
                raise VertexJobObservationError("Invalid US Vertex Custom Job locator")

    @property
    def resource_name(self) -> str:
        return (
            f"projects/{self.project_number}/locations/{self.region}/"
            f"customJobs/{self.job_id}"
        )

    @property
    def get_url(self) -> str:
        return f"https://{self.region}-aiplatform.googleapis.com/v1/{self.resource_name}"


class VertexCustomJobState(StrEnum):
    QUEUED = "JOB_STATE_QUEUED"
    PENDING = "JOB_STATE_PENDING"
    RUNNING = "JOB_STATE_RUNNING"
    SUCCEEDED = "JOB_STATE_SUCCEEDED"
    FAILED = "JOB_STATE_FAILED"
    CANCELLING = "JOB_STATE_CANCELLING"
    CANCELLED = "JOB_STATE_CANCELLED"
    PAUSED = "JOB_STATE_PAUSED"
    EXPIRED = "JOB_STATE_EXPIRED"
    UPDATING = "JOB_STATE_UPDATING"
    PARTIALLY_SUCCEEDED = "JOB_STATE_PARTIALLY_SUCCEEDED"


@dataclass(frozen=True, slots=True)
class VertexCustomJobObservation:
    """Provider timestamps and state, not artifact, cost, or cleanup authority."""

    locator: VertexCustomJobLocator
    operation_sha256: str
    state: VertexCustomJobState
    created_at: datetime
    started_at: datetime | None
    ended_at: datetime | None

    @classmethod
    def from_readback(
        cls,
        payload: Mapping[str, Any],
        *,
        locator: VertexCustomJobLocator,
        expected_operation_sha256: str,
    ) -> VertexCustomJobObservation:
        if not isinstance(expected_operation_sha256, str) or not _SHA256.fullmatch(
            expected_operation_sha256
        ):
            raise VertexJobObservationError("Invalid original Vertex operation digest")
        if not isinstance(payload, Mapping):
            raise VertexJobObservationError("Invalid Vertex readback object")
        labels = payload.get("labels")
        if (
            payload.get("name") != locator.resource_name
            or not isinstance(labels, Mapping)
            or labels.get(VERTEX_OPERATION_LABEL) != expected_operation_sha256
        ):
            raise VertexJobObservationError("Vertex readback does not bind the original job")

        try:
            state = VertexCustomJobState(payload.get("state"))
        except (TypeError, ValueError) as error:
            raise VertexJobObservationError("Unrecognized Vertex job state") from error
        created_at = _timestamp(payload.get("createTime"))
        started_at = _timestamp(payload.get("startTime"))
        ended_at = _timestamp(payload.get("endTime"))
        if created_at is None:
            raise VertexJobObservationError("Vertex readback lacks its create timestamp")
        if state in {
            VertexCustomJobState.RUNNING, VertexCustomJobState.UPDATING,
            VertexCustomJobState.SUCCEEDED,
        } and started_at is None:
            raise VertexJobObservationError("Vertex readback lacks its start timestamp")
        if state in {
            VertexCustomJobState.SUCCEEDED, VertexCustomJobState.FAILED,
            VertexCustomJobState.CANCELLED,
        } and ended_at is None:
            raise VertexJobObservationError("Vertex readback lacks its end timestamp")
        if (
            (started_at is not None and started_at < created_at)
            or (ended_at is not None and ended_at < (started_at or created_at))
        ):
            raise VertexJobObservationError("Vertex job timestamps are out of order")
        return cls(
            locator, expected_operation_sha256, state, created_at[0],
            started_at[0] if started_at is not None else None,
            ended_at[0] if ended_at is not None else None,
        )


def _timestamp(value: Any) -> tuple[datetime, int] | None:
    """Retain the sub-microsecond remainder so ordering does not round it away."""
    if value is None:
        return None
    match = _RFC3339.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise VertexJobObservationError("Invalid Vertex timestamp")
    try:
        timestamp = datetime.fromisoformat(value).astimezone(UTC)
    except (ValueError, OverflowError) as error:
        raise VertexJobObservationError("Invalid Vertex timestamp") from error
    fraction = (match["fraction"] or "").ljust(9, "0")
    return timestamp, int(fraction[6:])
