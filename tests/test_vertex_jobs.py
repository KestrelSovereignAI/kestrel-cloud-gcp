"""Vertex v1 Custom Job evidence and exact identity contract."""

from base64 import b32decode
from datetime import UTC, datetime

import pytest

from kestrel_cloud_gcp.vertex_jobs import (
    VERTEX_OPERATION_LABEL,
    VertexCustomJobLocator,
    VertexCustomJobObservation,
    VertexCustomJobState,
    VertexJobObservationError,
    vertex_operation_label_value,
)

JOB = VertexCustomJobLocator("123456789", "us-central1", "35291")
DIGEST = "a" * 64


def payload(**fields):
    return {
        "name": JOB.resource_name,
        "state": "JOB_STATE_SUCCEEDED",
        "createTime": "2026-11-22T17:00:00Z",
        "startTime": "2026-11-22T17:02:10Z",
        "endTime": "2026-11-22T10:08:10-07:00",
        "labels": {VERTEX_OPERATION_LABEL: vertex_operation_label_value(DIGEST)},
        **fields,
    }


def read(fields):
    return VertexCustomJobObservation.from_readback(
        fields, locator=JOB, expected_operation_sha256=DIGEST,
    )


def test_exact_job_url_and_provider_clock_readback():
    assert JOB.get_url == (
        "https://us-central1-aiplatform.googleapis.com/v1/"
        "projects/123456789/locations/us-central1/customJobs/35291"
    )
    observation = read(payload(jobSpec={"private": "never retained"}))
    assert observation.state is VertexCustomJobState.SUCCEEDED
    assert observation.ended_at == datetime(2026, 11, 22, 17, 8, 10, tzinfo=UTC)
    assert (observation.started_at - observation.created_at).total_seconds() == 130
    assert "never retained" not in repr(observation)


def test_cancelled_before_start_does_not_invent_a_start_or_success():
    observation = read(payload(state="JOB_STATE_CANCELLED", startTime=None))
    assert observation.state is VertexCustomJobState.CANCELLED
    assert observation.started_at is None


def test_original_digest_label_fits_and_preserves_every_digest_byte():
    label = vertex_operation_label_value(DIGEST)
    assert len(label) == 52
    assert label.islower()
    assert b32decode(label.upper() + "=" * (-len(label) % 8)).hex() == DIGEST


@pytest.mark.parametrize("state", ("succeeded", "failed", "cancelled", "expired", "partially_succeeded"))
def test_terminal_readback_requires_provider_end_evidence(state):
    with pytest.raises(VertexJobObservationError, match="lacks its end timestamp"):
        read(payload(state="JOB_STATE_" + state.upper(), endTime=None))


@pytest.mark.parametrize("state", ("running", "updating", "succeeded", "partially_succeeded"))
def test_active_or_successful_readback_requires_provider_start_evidence(state):
    with pytest.raises(VertexJobObservationError, match="lacks its start timestamp"):
        read(payload(state="JOB_STATE_" + state.upper(), startTime=None))


@pytest.mark.parametrize(
    "overrides",
    [
        {"name": "projects/123456789/locations/us-central1/customJobs/35292"},
        {"name": "projects/123456789/locations/northamerica-northeast1/customJobs/35291"},
        {"name": "projects/123456790/locations/us-central1/customJobs/35291"},
        {"labels": {VERTEX_OPERATION_LABEL: vertex_operation_label_value("b" * 64)}},
        {"labels": {VERTEX_OPERATION_LABEL: DIGEST}},
        {"labels": {}},
        {"state": "JOB_STATE_UNKNOWN"},
        {"createTime": None},
        {"startTime": None},
        {"endTime": None},
        {"createTime": "2026-11-22T17:00:00"},
        {"createTime": "2026-11-22T17:00:00+00:75"},
        {"startTime": "2026-11-22T16:59:59Z"},
        {"endTime": "2026-11-22T17:02:09Z"},
        {
            "startTime": "2026-11-22T17:02:10.001234999Z",
            "endTime": "2026-11-22T17:02:10.001234998Z",
        },
    ],
)
def test_refuses_foreign_job_or_unconfirmed_timeline(overrides):
    with pytest.raises(VertexJobObservationError):
        read(payload(**overrides))


@pytest.mark.parametrize(
    "address",
    [
        ("123456789", "northamerica-northeast1", "35291"),
        ("123456789", "us-central1/../../", "35291"),
        ("legacy-alias", "us-central1", "35291"),
        ("123456789", "us-central1", "abc/123"),
    ],
)
def test_only_explicit_us_job_resource_names_are_accepted(address):
    with pytest.raises(VertexJobObservationError):
        VertexCustomJobLocator(*address)
