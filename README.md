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
