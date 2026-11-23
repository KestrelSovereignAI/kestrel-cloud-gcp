# kestrel-cloud-gcp

GCP Compute Engine provider for Kestrel Sovereign agents. Provision GPU instances, run LoRA training jobs over SSH, and manage instance lifecycle from your agent's tools.

## Installation

```bash
uv pip install kestrel-cloud-gcp
```

The feature is auto-discovered by Kestrel Sovereign via the `kestrel_sovereign.features` entry point — install it alongside `kestrel-sovereign` and `GCPComputeFeature` registers itself at startup.

## Configuration

| Variable | Description |
|----------|-------------|
| `GCP_PROJECT_ID` | GCP project ID (required) |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to service-account JSON (or use ADC / Workload Identity) |

Optional `[gcp_compute]` section in `kestrel.toml`:

```toml
[gcp_compute.manager]
default_zone = "us-central1-a"
default_region = "us-central1"
prefer_spot = true
```

## What's provided

- `GCPComputeFeature` — agent-facing tools to provision/list/stop GPU instances, run training over SSH, monitor jobs
- Standalone API: `GCPComputeEngineManager` for direct programmatic use
- Spot-instance preference + automatic lifecycle (TTL, budget caps)

### Vertex Custom Job admission and recovery

Starting in `kestrel-cloud-gcp` 0.3.0, `kestrel_cloud_gcp.vertex_jobs` provides
US Vertex AI v1 Custom Job submission and recovery. Bind the original dispatch
digest with `vertex_operation_label_value`
and use `find_vertex_custom_job` with an authenticated GET callback to recover
the exact job after a lost submit response. The search rejects multiple matches,
foreign identities, and incomplete pages. An empty result never authorizes a
repeat submission. Verify subsequent status with the exact job URL and
`VertexCustomJobObservation.from_readback`.

`VertexCustomJobClient.from_adc()` uses Google Application Default Credentials
to submit an already-persisted `VertexCustomJobPlan` through the regional v1
API. It does not replay a create, follow redirects, or treat an acknowledged
cancel as proof. Persist the exact job identity or recover it by operation
label; cancel only that identity and verify terminal state through readback.
The caller owns spend admission, queue and execution deadlines, reconciliation,
and catalog artifact custody. The adapter does not retain trainer arguments,
outputs, or raw provider error bodies.

`VertexCustomJobPlan` prepares the US v1 request for one GPU container
without sending it. The image must be digest-pinned in regional Artifact
Registry, the worker must have an explicit workload service account, and
the runtime limit and last submission time must fit its admission deadline.
The runtime limit does not bound time spent queued; the caller must own
absolute-deadline cancellation. This class does not call create or cancel.
The container receives one derived environment value,
`KESTREL_VERTEX_OPERATION_SHA256`, matching the original digest encoded in
the management label. It is a correlation key only, never proof of identity
or authorization to fetch artifacts. The job plan accepts no ambient worker
environment, custom arguments, or user artifact credentials.

## Dependencies

- `kestrel-sovereign-sdk>=0.2,<1` — base `Feature`, `tool`, `ToolCategory`, `BackendType` interfaces
- `kestrel-sovereign>=0.5,<1` — `kestrel.toml` unified-config loader (runtime)
- `google-cloud-compute>=1.15.0`
- `google-cloud-logging>=3.5.0`

## Development

```bash
uv pip install -e '.[test]'
uv run pytest
```

PyPI releases use an immutable version tag on `main`. Tag CI verifies the
version and installs the same wheel that it hands to the trusted-publishing
job, together with its source distribution. It does not rebuild after CI.

## License

Apache-2.0
