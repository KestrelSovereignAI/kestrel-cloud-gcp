"""Content-free readback and recovery evidence for Vertex v1 Custom Jobs."""

from __future__ import annotations

import re
from base64 import b32encode
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

VERTEX_OPERATION_LABEL = "kestrel_operation_sha256"
_NUMBER = re.compile(r"[1-9][0-9]{0,19}")
_US_REGION = re.compile(r"us-[a-z]+[0-9]+")
_SHA256 = re.compile(r"[a-f0-9]{64}")
_MACHINE = re.compile(r"[a-z][a-z0-9-]{1,62}")
# Reviewed single-worker options from Vertex v1 AcceleratorType; machine
# compatibility and quota remain an operator/catalog admission check.
_ACCELERATORS = frozenset({"NVIDIA_TESLA_T4", "NVIDIA_L4", "NVIDIA_TESLA_A100", "NVIDIA_A100_80GB"})
_SERVICE_ACCOUNT = re.compile(
    r"[a-z][a-z0-9-]{4,28}[a-z0-9]@[a-z][a-z0-9-]{4,28}[a-z0-9]\.iam\.gserviceaccount\.com"
)
_IMAGE_PATH = re.compile(
    r"[a-z][a-z0-9-]{4,28}[a-z0-9]/[a-z0-9][a-z0-9_-]*/"
    r"(?:[a-z0-9][a-z0-9_.-]*/)*[a-z0-9][a-z0-9_.-]*@sha256:[a-f0-9]{64}"
)
_RFC3339 = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.(?P<fraction>[0-9]{1,9}))?(?:Z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])"
)


class VertexJobObservationError(ValueError):
    """The provider observation cannot establish the original job's state."""


@dataclass(frozen=True, slots=True)
class VertexCustomJobLocation:
    """Explicit US Vertex Custom Job collection, with no project alias."""

    project_number: str
    region: str

    def __post_init__(self) -> None:
        for value, pattern in (
            (self.project_number, _NUMBER),
            (self.region, _US_REGION),
        ):
            if not isinstance(value, str) or not pattern.fullmatch(value):
                raise VertexJobObservationError("Invalid US Vertex Custom Job locator")

    @property
    def collection_name(self) -> str:
        return f"projects/{self.project_number}/locations/{self.region}/customJobs"

    @property
    def list_url(self) -> str:
        return f"https://{self.region}-aiplatform.googleapis.com/v1/{self.collection_name}"

    def identify(self, resource_name: object) -> VertexCustomJobLocator:
        prefix = self.collection_name + "/"
        if not isinstance(resource_name, str) or not resource_name.startswith(prefix):
            raise VertexJobObservationError("Vertex search returned a foreign job")
        return VertexCustomJobLocator(self.project_number, self.region, resource_name[len(prefix):])


@dataclass(frozen=True, slots=True)
class VertexCustomJobLocator(VertexCustomJobLocation):
    """Exact resource name bound to a configured Google project number."""

    job_id: str

    def __post_init__(self) -> None:
        VertexCustomJobLocation.__post_init__(self)
        if not isinstance(self.job_id, str) or not _NUMBER.fullmatch(self.job_id):
            raise VertexJobObservationError("Invalid US Vertex Custom Job locator")

    @property
    def resource_name(self) -> str:
        return f"{self.collection_name}/{self.job_id}"

    @property
    def get_url(self) -> str:
        return f"{self.list_url}/{self.job_id}"


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


_START_EVIDENCE_REQUIRED = frozenset({
    VertexCustomJobState.RUNNING,
    VertexCustomJobState.UPDATING,
    VertexCustomJobState.SUCCEEDED,
    VertexCustomJobState.PARTIALLY_SUCCEEDED,
})
_END_EVIDENCE_REQUIRED = frozenset({
    VertexCustomJobState.SUCCEEDED,
    VertexCustomJobState.FAILED,
    VertexCustomJobState.CANCELLED,
    VertexCustomJobState.EXPIRED,
    VertexCustomJobState.PARTIALLY_SUCCEEDED,
})


def vertex_operation_label_value(operation_sha256: str) -> str:
    """Losslessly encode a digest below Google's 63-character label limit."""
    if not isinstance(operation_sha256, str) or not _SHA256.fullmatch(operation_sha256):
        raise VertexJobObservationError("Invalid original Vertex operation digest")
    return b32encode(bytes.fromhex(operation_sha256)).decode("ascii").rstrip("=").lower()


@dataclass(frozen=True, slots=True)
class VertexCustomJobPlan:
    """A single regional GPU container plan, not authorization to submit it."""

    location: VertexCustomJobLocation
    operation_sha256: str
    worker_service_account: str
    image_uri: str
    machine_type: str
    accelerator_type: str
    max_run_seconds: int
    submit_before: datetime
    hard_deadline: datetime

    def __post_init__(self) -> None:
        vertex_operation_label_value(self.operation_sha256)
        if not isinstance(self.location, VertexCustomJobLocation) or isinstance(
            self.location, VertexCustomJobLocator
        ):
            raise VertexJobObservationError("Vertex plan requires a collection location")
        prefix = f"{self.location.region}-docker.pkg.dev/"
        if (
            not isinstance(self.image_uri, str) or not self.image_uri.startswith(prefix)
            or _IMAGE_PATH.fullmatch(self.image_uri[len(prefix):]) is None
        ):
            raise VertexJobObservationError("Vertex plan requires a colocated digest-pinned image")
        for value, pattern in (
            (self.worker_service_account, _SERVICE_ACCOUNT),
            (self.machine_type, _MACHINE),
        ):
            if not isinstance(value, str) or pattern.fullmatch(value) is None:
                raise VertexJobObservationError("Invalid Vertex workload identity or machine shape")
        if not isinstance(self.accelerator_type, str) or self.accelerator_type not in _ACCELERATORS:
            raise VertexJobObservationError("Vertex GPU type is outside the reviewed v1 set")
        if type(self.max_run_seconds) is not int or self.max_run_seconds < 1:
            raise VertexJobObservationError("Vertex job requires a positive runtime limit")
        for name in ("submit_before", "hard_deadline"):
            deadline = getattr(self, name)
            if not isinstance(deadline, datetime) or deadline.utcoffset() is None:
                raise VertexJobObservationError("Vertex plan deadlines must be timezone-aware")
            object.__setattr__(self, name, deadline.astimezone(UTC))
        if (self.hard_deadline - self.submit_before).total_seconds() < self.max_run_seconds:
            raise VertexJobObservationError("Vertex runtime cannot fit into the original admission")

    @property
    def create_url(self) -> str:
        return self.location.list_url

    def create_payload(self, *, now: datetime) -> dict[str, Any]:
        """Encode a request before POST without sending it or retaining secrets.

        Vertex's runtime timeout does not bound queue time. The caller remains
        responsible for the absolute deadline, including cancellation in queue.
        """
        if (
            not isinstance(now, datetime) or now.utcoffset() is None
            or now.astimezone(UTC) >= self.submit_before
        ):
            raise VertexJobObservationError("Vertex submission window has elapsed")
        label = vertex_operation_label_value(self.operation_sha256)
        return {
            "displayName": "kestrel-" + label,
            "labels": {VERTEX_OPERATION_LABEL: label},
            "jobSpec": {
                "serviceAccount": self.worker_service_account,
                "enableWebAccess": False,
                "enableDashboardAccess": False,
                "scheduling": {
                    "timeout": f"{self.max_run_seconds}s", "disableRetries": True,
                    "restartJobOnWorkerRestart": False,
                },
                "workerPoolSpecs": [{
                    "replicaCount": "1",
                    "machineSpec": {
                        "machineType": self.machine_type,
                        "acceleratorType": self.accelerator_type,
                        "acceleratorCount": 1,
                    },
                    "containerSpec": {"imageUri": self.image_uri},
                }],
            },
        }

    def observe_create_response(self, payload: Mapping[str, Any]) -> VertexCustomJobObservation:
        """Bind one ack; a bad/missing ack is unresolved, never safe to replay."""
        if not isinstance(payload, Mapping):
            raise VertexJobObservationError("Invalid Vertex create response")
        return VertexCustomJobObservation.from_readback(
            payload, locator=self.location.identify(payload.get("name")),
            expected_operation_sha256=self.operation_sha256,
        )


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
        expected_label = vertex_operation_label_value(expected_operation_sha256)
        if not isinstance(payload, Mapping):
            raise VertexJobObservationError("Invalid Vertex readback object")
        labels = payload.get("labels")
        if (
            payload.get("name") != locator.resource_name
            or not isinstance(labels, Mapping)
            or labels.get(VERTEX_OPERATION_LABEL) != expected_label
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
        if state in _START_EVIDENCE_REQUIRED and started_at is None:
            raise VertexJobObservationError("Vertex readback lacks its start timestamp")
        if state in _END_EVIDENCE_REQUIRED and ended_at is None:
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


def find_vertex_custom_job(
    fetch_page: Callable[[str, Mapping[str, str]], Mapping[str, Any]],
    *,
    location: VertexCustomJobLocation,
    operation_sha256: str,
    max_pages: int = 20,
) -> VertexCustomJobObservation | None:
    """Find one exact advertised job through the read-only v1 label filter.

    The authenticated transport is supplied by the caller. A complete empty
    result describes this observation only, never permission to retry a create
    after an ambiguous result. Multiple matches or incomplete evidence fail
    closed. A caller must persist the confirmed job identity before effects.
    """
    if type(max_pages) is not int or max_pages < 1:
        raise ValueError("Vertex search requires a positive page bound")
    label = vertex_operation_label_value(operation_sha256)
    query = {
        "filter": f"labels.{VERTEX_OPERATION_LABEL}={label}",
        "fields": "customJobs(name,labels,state,createTime,startTime,endTime),nextPageToken",
    }
    candidate: VertexCustomJobObservation | None = None
    seen_tokens: set[str] = set()
    for _ in range(max_pages):
        payload = fetch_page(location.list_url, dict(query))
        if not isinstance(payload, Mapping) or set(payload).difference(
            {"customJobs", "nextPageToken"}
        ):
            raise VertexJobObservationError("Invalid Vertex job list response")
        rows, token = payload.get("customJobs", []), payload.get("nextPageToken", "")
        if not isinstance(rows, list) or not isinstance(token, str):
            raise VertexJobObservationError("Invalid Vertex job list response")
        for row in rows:
            if not isinstance(row, Mapping):
                raise VertexJobObservationError("Invalid Vertex job list record")
            observation = VertexCustomJobObservation.from_readback(
                row,
                locator=location.identify(row.get("name")),
                expected_operation_sha256=operation_sha256,
            )
            if candidate is not None:
                raise VertexJobObservationError("Vertex search found multiple observations")
            candidate = observation
        if not token:
            return candidate
        if token in seen_tokens:
            raise VertexJobObservationError("Vertex search repeated a page token")
        seen_tokens.add(token)
        query["pageToken"] = token
    raise VertexJobObservationError("Vertex search did not complete within its page bound")


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
