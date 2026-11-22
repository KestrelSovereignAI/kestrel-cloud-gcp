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

### Vertex Custom Job recovery

`kestrel_cloud_gcp.vertex_jobs` offers read-only helpers for US Vertex AI v1
Custom Jobs. Bind the original dispatch digest with `vertex_operation_label_value`
and use `find_vertex_custom_job` with an authenticated GET callback to recover
the exact job after a lost submit response. The search rejects multiple matches,
foreign identities, and incomplete pages. An empty result never authorizes a
repeat submission. Verify subsequent status with the exact job URL and
`VertexCustomJobObservation.from_readback`. The helpers do not submit or cancel
jobs and never retain trainer arguments or artifacts.

`VertexCustomJobPlan` prepares the US v1 request for one GPU container
without sending it. The image must be digest-pinned in regional Artifact
Registry, the worker must have an explicit workload service account, and
the runtime limit and last submission time must fit its admission deadline.
The runtime limit does not bound time spent queued; the caller must own
absolute-deadline cancellation. This class does not call create or cancel.

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

## License

Apache-2.0
