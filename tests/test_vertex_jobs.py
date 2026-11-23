"""Vertex v1 Custom Job evidence and exact identity contract."""

from base64 import b32decode
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import json
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread
from urllib.parse import urlsplit

import pytest
from google.auth.credentials import Credentials
from requests.adapters import HTTPAdapter

from kestrel_cloud_gcp.vertex_jobs import (
    VERTEX_OPERATION_ENV,
    VERTEX_OPERATION_LABEL,
    VertexCustomJobClient,
    VertexCustomJobLocation,
    VertexCustomJobLocator,
    VertexCustomJobObservation,
    VertexCustomJobPlan,
    VertexCustomJobState,
    VertexJobObservationError,
    VertexJobEffectUnresolvedError,
    VertexWorkerDispatch,
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


@pytest.fixture
def bounded_plan():
    now = datetime(2026, 11, 22, 16, 59, tzinfo=UTC)
    return VertexCustomJobPlan(
        location=LOCATION, operation_sha256=DIGEST,
        worker_service_account="lora-worker@training-project.iam.gserviceaccount.com",
        image_uri="us-central1-docker.pkg.dev/training-project/worker/sdxl@sha256:" + "a" * 64,
        machine_type="g2-standard-8", accelerator_type="NVIDIA_L4",
        max_run_seconds=1800, submit_before=now + timedelta(seconds=60),
        hard_deadline=now + timedelta(seconds=3600),
    )


@pytest.fixture
def vertex_wire():
    expected = deque()
    calls = []

    class ProviderHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def receive(self):
            request_data = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            assert self.headers.get("Authorization") == "Bearer loopback-integration-only"
            calls.append((
                self.command, self.path, json.loads(request_data) if request_data else None,
            ))
            status, payload, headers = expected.popleft()
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        do_GET = do_POST = receive

    class Credential(Credentials):
        def __init__(self):
            super().__init__()
            self.token = "loopback-integration-only"
            self.refresh_calls = 0
            self.refresh_delay = 0

        def refresh(self, request):
            self.refresh_calls += 1
            Event().wait(self.refresh_delay)
            self.token = "loopback-integration-only"

    with ThreadingHTTPServer(("127.0.0.1", 0), ProviderHandler) as server:
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        thread.start()

        class Loopback(HTTPAdapter):
            def send(self, request, **kwargs):
                original = urlsplit(request.url)
                assert original.netloc == "us-central1-aiplatform.googleapis.com"
                request.url = f"http://127.0.0.1:{server.server_port}{original.path}"
                if original.query:
                    request.url += "?" + original.query
                return super().send(request, **kwargs)

        def respond(status, payload, **headers):
            expected.append((status, payload, headers))

        credential = Credential()
        try:
            with VertexCustomJobClient(credential) as client:
                client._session.mount("https://us-central1-aiplatform.googleapis.com/", Loopback())
                yield client, respond, calls, credential
                assert not expected, "The adapter skipped an expected exact readback"
        finally:
            server.shutdown()
            thread.join(timeout=1)


def test_worker_rendezvous_uses_only_fixed_nonsecret_environment(vertex_wire, bounded_plan):
    client, respond, calls, _ = vertex_wire
    plan = replace(bounded_plan, worker_dispatch=VertexWorkerDispatch("private-test-bucket", "vault.example.test"))
    spec = plan.create_payload(now=plan.submit_before - timedelta(seconds=1))["jobSpec"]
    env = spec["workerPoolSpecs"][0]["containerSpec"]["env"]
    assert dict((item["name"], item["value"]) for item in env) == {
        VERTEX_OPERATION_ENV: DIGEST,
        "CATALOG_ARTIFACT_HOSTS": "storage.googleapis.com,vault.example.test",
        "CONTAINER_DIGEST": "sha256:" + "a" * 64,
        "KESTREL_VERTEX_DISPATCH_BUCKET": "private-test-bucket",
        "KESTREL_VERTEX_HARD_DEADLINE": plan.hard_deadline.isoformat(),
        "KESTREL_VERTEX_JOB_COLLECTION": LOCATION.collection_name,
    }
    respond(200, payload(jobSpec=spec))
    assert client.get_for_worker_dispatch(JOB, plan) == read(payload())
    assert calls[0][0] == "GET"
    assert "workerPoolSpecs" in calls[0][1]


@pytest.mark.parametrize("drift", [
    "user", "gpu", "image", "count", "command", "env", "retries", "persistent",
    "network", "reserved", "psc", "strategy",
])
def test_refuses_worker_release_if_exact_deployed_spec_drifts(bounded_plan, drift):
    plan = replace(bounded_plan, worker_dispatch=VertexWorkerDispatch("private-test-bucket", "vault.example.test"))
    spec = plan.create_payload(now=plan.submit_before - timedelta(seconds=1))["jobSpec"]
    pool = spec["workerPoolSpecs"][0]
    if drift == "user":
        spec["serviceAccount"] = "wrong@training-project.iam.gserviceaccount.com"
    elif drift == "persistent":
        spec["persistentResourceId"] = "foreign"
    elif drift == "network":
        spec["network"] = "projects/123/global/networks/foreign"
    elif drift == "reserved":
        spec["reservedIpRanges"] = ["foreign"]
    elif drift == "psc":
        spec["pscInterfaceConfig"] = {"networkAttachment": "foreign"}
    elif drift == "strategy":
        spec["scheduling"]["strategy"] = "SPOT"
    elif drift == "gpu":
        pool["machineSpec"]["acceleratorType"] = "NVIDIA_TESLA_A100"
    elif drift == "count":
        pool["replicaCount"] = "2"
    elif drift == "retries":
        spec["scheduling"]["disableRetries"] = False
    elif drift == "image":
        pool["containerSpec"]["imageUri"] = "different"
    elif drift == "command":
        pool["containerSpec"]["command"] = ["python", "other.py"]
    else:
        pool["containerSpec"]["env"].append({"name": "not-approved", "value": "no"})
    with pytest.raises(VertexJobObservationError, match="changed"):
        plan.verify_worker_readback(payload(jobSpec=spec), JOB)


@pytest.mark.parametrize("bucket,host", [
    ("gs://foreign", "vault.example.test"),
    ("valid-bucket", "https://vault.example.test"),
    ("valid-bucket", "user@vault.example.test"),
    ("valid-bucket", "127.0.0.1"),
])
def test_rejects_untrusted_dispatch_location_shapes(bucket, host):
    with pytest.raises(VertexJobObservationError, match="dispatch location"):
        VertexWorkerDispatch(bucket, host)


@pytest.mark.parametrize("status", [200, 302, 401, 503])
def test_vertex_create_sends_once_and_recovers_only_by_label(vertex_wire, bounded_plan, status):
    client, respond, calls, credential = vertex_wire
    now = datetime.now(UTC)
    plan = replace(
        bounded_plan, submit_before=now + timedelta(minutes=1),
        hard_deadline=now + timedelta(hours=1),
    )
    respond(status, payload(), Location=JOB.get_url)
    if status == 200:
        assert client.create(plan).locator == JOB
    else:
        with pytest.raises(VertexJobEffectUnresolvedError, match="original operation label"):
            client.create(plan)
        respond(200, {})
        assert client.find(plan.location, plan.operation_sha256) is None
        assert [m for m, *_ in calls] == ["POST", "GET"]
    assert calls[0][:2] == ("POST", "/v1/" + LOCATION.collection_name)
    assert calls[0][2] == plan.create_payload(now=now)
    assert credential.refresh_calls == 0


def test_expired_vertex_plan_reaches_no_transport(vertex_wire, bounded_plan):
    client, _, calls, _ = vertex_wire
    now = datetime.now(UTC)
    elapsed = replace(
        bounded_plan, submit_before=now - timedelta(seconds=1),
        hard_deadline=now + timedelta(hours=1),
    )
    with pytest.raises(VertexJobObservationError, match="window has elapsed"):
        client.create(elapsed)
    assert not calls


def test_vertex_plan_rechecks_deadline_after_google_credential_refresh(vertex_wire, bounded_plan):
    client, _, calls, credential = vertex_wire
    now = datetime.now(UTC)
    plan = replace(
        bounded_plan, submit_before=now + timedelta(milliseconds=100),
        hard_deadline=now + timedelta(hours=1),
    )
    credential.token = None
    credential.refresh_delay = 0.15
    with pytest.raises(VertexJobObservationError, match="window elapsed before transport") as info:
        client.create(plan)
    assert not isinstance(info.value, VertexJobEffectUnresolvedError)
    assert credential.refresh_calls == 1
    assert not calls


@pytest.mark.parametrize("post_status", [200, 503])
@pytest.mark.parametrize("final_state", ["CANCELLED", "SUCCEEDED", "CANCELLING"])
def test_vertex_cancel_requires_exact_final_readback(vertex_wire, post_status, final_state):
    client, respond, calls, _ = vertex_wire
    respond(200, payload(state="JOB_STATE_RUNNING", endTime=None))
    respond(post_status, {})
    end = None if final_state == "CANCELLING" else "2026-11-22T17:06:30Z"
    respond(200, payload(state="JOB_STATE_" + final_state, endTime=end))
    if end is None:
        with pytest.raises(VertexJobEffectUnresolvedError, match="no exact terminal readback"):
            client.cancel(JOB, DIGEST)
    else:
        assert client.cancel(JOB, DIGEST).state is VertexCustomJobState[final_state]
    assert [method for method, *_ in calls] == ["GET", "POST", "GET"]
    assert calls[1] == ("POST", f"/v1/{JOB.resource_name}:cancel", {})
    assert all(urlsplit(path).path == f"/v1/{JOB.resource_name}" for _, path, _ in calls[::2])


@pytest.mark.parametrize("terminal", [True, False])
def test_vertex_cancel_skips_terminal_jobs_and_refuses_foreign_readback(vertex_wire, terminal):
    client, respond, calls, _ = vertex_wire
    if terminal:
        respond(200, payload())
        assert client.cancel(JOB, DIGEST).state is VertexCustomJobState.SUCCEEDED
    else:
        respond(200, payload(labels={}))
        with pytest.raises(VertexJobObservationError, match="does not bind the original job"):
            client.cancel(JOB, DIGEST)
    assert len(calls) == 1 and calls[0][0] == "GET"


def test_worker_operation_correlation_matches_the_original_management_label(bounded_plan):
    original = "0123456789abcdef" * 4
    plan = replace(bounded_plan, operation_sha256=original)
    now = plan.submit_before - timedelta(seconds=1)
    payload = plan.create_payload(now=now)
    env = payload["jobSpec"]["workerPoolSpecs"][0]["containerSpec"]["env"]
    label = payload["labels"][VERTEX_OPERATION_LABEL]
    assert env == [{"name": VERTEX_OPERATION_ENV, "value": original}]
    assert b32decode(label.upper() + "=" * (-len(label) % 8)).hex() == env[0]["value"]
    env[0]["value"] = "mutated"
    assert plan.create_payload(now=now)["jobSpec"]["workerPoolSpecs"][0]["containerSpec"]["env"][0]["value"] == original


def test_create_payload_has_one_pinned_worker_explicit_deadline_and_no_ambient_credentials(bounded_plan):
    now = bounded_plan.submit_before - timedelta(seconds=1)
    wire = bounded_plan.create_payload(now=now)
    assert bounded_plan.create_url == LOCATION.list_url
    assert wire["labels"] == {VERTEX_OPERATION_LABEL: vertex_operation_label_value(DIGEST)}
    assert wire["jobSpec"] == {
        "serviceAccount": bounded_plan.worker_service_account,
        "enableWebAccess": False, "enableDashboardAccess": False,
        "scheduling": {
            "timeout": "1800s", "disableRetries": True, "restartJobOnWorkerRestart": False,
        },
        "workerPoolSpecs": [{
            "replicaCount": "1",
            "machineSpec": {
                "machineType": "g2-standard-8", "acceleratorType": "NVIDIA_L4",
                "acceleratorCount": 1,
            },
            "containerSpec": {
                "imageUri": bounded_plan.image_uri,
                "env": [{"name": "KESTREL_VERTEX_OPERATION_SHA256", "value": DIGEST}],
            },
        }],
    }
    wire["jobSpec"]["workerPoolSpecs"].clear()
    assert len(bounded_plan.create_payload(now=now)["jobSpec"]["workerPoolSpecs"]) == 1


def test_plan_refuses_late_submit_and_binds_ack_to_original_operation(bounded_plan):
    with pytest.raises(VertexJobObservationError, match="window has elapsed"):
        bounded_plan.create_payload(now=bounded_plan.submit_before)
    assert bounded_plan.observe_create_response(payload()) == read(payload())
    for foreign in (payload(labels={}), payload(name="projects/2/locations/us-east1/customJobs/1")):
        with pytest.raises(VertexJobObservationError):
            bounded_plan.observe_create_response(foreign)


@pytest.mark.parametrize(
    "changes",
    [
        {"location": JOB},
        {"image_uri": "us-central1-docker.pkg.dev/training-project/worker/sdxl:latest"},
        {"image_uri": "europe-west3-docker.pkg.dev/training-project/worker/sdxl@sha256:" + "a" * 64},
        {"worker_service_account": "person@example.test"},
        {"max_run_seconds": True},
        {"max_run_seconds": 0},
        {"max_run_seconds": 5000},
        {"accelerator_type": "TPU_PAUSE"},
        {"accelerator_type": "NVIDIA_TYPO"},
    ],
)
def test_invalid_vertex_shape_cannot_produce_a_create_plan(bounded_plan, changes):
    with pytest.raises(VertexJobObservationError):
        replace(bounded_plan, **changes)


def test_vertex_deadlines_use_elapsed_time_across_both_daylight_saving_changes(bounded_plan):
    denver = ZoneInfo("America/Denver")
    with pytest.raises(VertexJobObservationError, match="cannot fit"):
        replace(
            bounded_plan,
            submit_before=datetime(2026, 3, 8, 1, 30, tzinfo=denver),
            hard_deadline=datetime(2026, 3, 8, 3, 30, tzinfo=denver),
            max_run_seconds=5400,
        )
    repeated_hour = replace(
        bounded_plan,
        submit_before=datetime(2026, 11, 1, 1, 40, tzinfo=denver, fold=0),
        hard_deadline=datetime(2026, 11, 1, 3, tzinfo=denver),
    )
    with pytest.raises(VertexJobObservationError, match="window has elapsed"):
        repeated_hour.create_payload(now=datetime(2026, 11, 1, 1, 10, tzinfo=denver, fold=1))
