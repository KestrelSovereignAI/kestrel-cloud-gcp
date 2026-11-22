"""Vertex v1 Custom Job evidence and exact identity contract."""

from base64 import b32decode
from datetime import UTC, datetime

import pytest

from kestrel_cloud_gcp.vertex_jobs import (
    VERTEX_OPERATION_LABEL,
    VertexCustomJobLocation,
    VertexCustomJobLocator,
    VertexCustomJobObservation,
    VertexCustomJobState,
    VertexJobObservationError,
    find_vertex_custom_job,
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


LOCATION = VertexCustomJobLocation(JOB.project_number, JOB.region)


def test_search_uses_only_exact_regional_list_and_paginates_after_a_match():
    calls = []
    pages = iter([
        {"customJobs": [payload()], "nextPageToken": " opaque-token "},
        {},
    ])

    def next_page(url, params):
        calls.append((url, params))
        return next(pages)

    assert find_vertex_custom_job(
        next_page, location=LOCATION, operation_sha256=DIGEST
    ) == read(payload())
    initial_query = {
        "filter": f"labels.{VERTEX_OPERATION_LABEL}={vertex_operation_label_value(DIGEST)}",
        "fields": "customJobs(name,labels,state,createTime,startTime,endTime),nextPageToken",
    }
    regional_url = JOB.get_url.rsplit("/", 1)[0]
    assert calls == [
        (regional_url, initial_query),
        (regional_url, {**initial_query, "pageToken": " opaque-token "}),
    ]


def test_complete_empty_vertex_search_is_not_a_job_observation():
    assert find_vertex_custom_job(
        lambda *_: {}, location=LOCATION, operation_sha256=DIGEST
    ) is None


@pytest.mark.parametrize(
    "pages, limit, problem",
    [
        ([{"customJobs": [payload(), payload()]}], 1, "multiple observations"),
        (
            [
                {"customJobs": [payload()], "nextPageToken": "token"},
                {"customJobs": [payload(name=JOB.resource_name[:-1] + "2")]},
            ],
            2,
            "multiple observations",
        ),
        ([{"nextPageToken": "token"}] * 2, 3, "repeated a page token"),
        ([{"customJobs": [payload()], "nextPageToken": "token"}], 1, "page bound"),
        ([{"nextPageToken": None}], 1, "Invalid Vertex job list response"),
        ([{"customJobs": [False]}], 1, "Invalid Vertex job list record"),
        ([{"customJobs": {}}], 1, "Invalid Vertex job list response"),
        ([{"error": {}}], 1, "Invalid Vertex job list response"),
        (
            [{"customJobs": [payload(name=JOB.resource_name.replace("us-", "europe-"))]}],
            1,
            "foreign job",
        ),
        ([{"customJobs": [payload(labels={})]}], 1, "does not bind the original job"),
    ],
)
def test_search_does_not_reattach_from_ambiguous_or_unverified_results(pages, limit, problem):
    responses = iter(pages)
    with pytest.raises(VertexJobObservationError, match=problem):
        find_vertex_custom_job(
            lambda *_: next(responses),
            location=LOCATION,
            operation_sha256=DIGEST,
            max_pages=limit,
        )


def test_vertex_search_rejects_invalid_bounds_before_transport():
    def forbidden(*_):
        pytest.fail("Search cannot run with an invalid bound")

    with pytest.raises(ValueError, match="positive page bound"):
        find_vertex_custom_job(
            forbidden, location=LOCATION, operation_sha256=DIGEST, max_pages=True
        )
