"""
Generates Markdown Cloud Runbooks for Hybrid RAG Retrieval & Citation Grounding.
"""

from pathlib import Path
from typing import Dict, List

CODE_FENCE = "```"


def _format_runbook(
    runbook_id: str,
    title: str,
    error_code: str,
    log_pattern: str,
    drivers: str,
    root_cause: str,
    commands: List[str],
) -> str:
    """Formats a structured runbook dictionary into a clean Markdown document."""
    cmd_blocks = []
    for idx, cmd in enumerate(commands, start=1):
        cmd_blocks.append(f"{idx}. Verified Command {idx}:\n{CODE_FENCE}bash\n{cmd}\n{CODE_FENCE}")
    joined_cmds = "\n".join(cmd_blocks)

    return (
        f"# Runbook ID: {runbook_id}\n"
        f"## Title: {title}\n"
        f"### Symptom Signature\n"
        f"- Error Code: `{error_code}`\n"
        f"- Log Pattern: `{log_pattern}`\n"
        f"- Correlated SHAP Drivers: {drivers}\n\n"
        f"### Root Cause Analysis\n"
        f"{root_cause}\n\n"
        f"### Verified Remediation Commands\n"
        f"{joined_cmds}\n"
    )


RUNBOOK_SPECS: List[Dict[str, object]] = [
    {
        "runbook_id": "RUNBOOK-K8S-137",
        "title": "GKE Container OOMKilled (Exit Code 137) on Inference Workers",
        "error_code": "K8S_OOM_KILLED",
        "log_pattern": "Container memory limit exceeded (exit code 137) on pod inference-worker",
        "drivers": "`k8s_oom_events_24h`, `quota_utilization_pct`, `p99_latency_ms`",
        "root_cause": (
            "Batch payload spikes during concurrent inference requests exceed pod "
            "cgroup memory ceiling, causing Linux kernel OOM killer termination."
        ),
        "commands": [
            (
                "kubectl set resources deployment inference-worker "
                "-c=inference-worker --limits=memory=16Gi --requests=memory=12Gi"
            ),
            "kubectl autoscale deployment inference-worker --cpu-percent=75 --min=3 --max=15",
        ],
    },
    {
        "runbook_id": "RUNBOOK-SQL-53300",
        "title": "Cloud SQL PostgreSQL Connection Pool Exhaustion (SQLSTATE 53300)",
        "error_code": "CLOUD_SQL_POOL_EXHAUSTED",
        "log_pattern": (
            "FATAL: remaining connection slots are reserved for "
            "non-replication superuser connections"
        ),
        "drivers": "`db_conn_pool_saturation_pct`, `p99_latency_ms`, `api_5xx_error_rate_pct`",
        "root_cause": (
            "Microservice instances open unpooled database connections without "
            "PgBouncer multiplexing, saturating max_connections during traffic bursts."
        ),
        "commands": [
            "gcloud sql instances patch prod-pg-instance --database-flags=max_connections=500",
            "export DB_POOL_SIZE=20 DB_MAX_OVERFLOW=10 DB_POOL_RECYCLE_SECONDS=1800",
        ],
    },
    {
        "runbook_id": "RUNBOOK-GPU-429",
        "title": "Regional NVIDIA A100 GPU Quota Throttling (ResourceExhausted 429)",
        "error_code": "GPU_QUOTA_THROTTLING",
        "log_pattern": "ResourceExhausted: Quota 'NVIDIA_A100_GPUS' exceeded in region us-central1",
        "drivers": "`gpu_throttle_ratio`, `quota_utilization_pct`",
        "root_cause": (
            "Autoscaling GPU node pools in a single region hit hard regional "
            "project quota limits during peak inference windows."
        ),
        "commands": [
            (
                "gcloud compute regions describe us-central1 "
                "--format='table(quotas:filter=metric:NVIDIA_A100_GPUS)'"
            ),
            (
                "gcloud container node-pools update gpu-pool-a100 "
                "--enable-autoscaling --max-nodes=24 --region=us-central1"
            ),
        ],
    },
    {
        "runbook_id": "RUNBOOK-GW-504",
        "title": "Cloud Load Balancing & API Gateway 5xx Upstream Timeout Spike",
        "error_code": "API_GATEWAY_5XX_SPIKE",
        "log_pattern": (
            "UpstreamTimeoutError: Back-end service failed to respond within 5000ms SLA window"
        ),
        "drivers": (
            "`api_5xx_error_rate_pct`, `support_ticket_velocity_7d`, "
            "`days_since_last_arch_review`"
        ),
        "root_cause": (
            "Keep-alive timeout mismatch between Cloud Load Balancer and upstream "
            "FastAPI workers causes premature 502/504 connection drops."
        ),
        "commands": [
            "gcloud compute backend-services update prod-api-backend --timeout=60s --global",
            (
                "gunicorn src.api.main:app -k uvicorn.workers.UvicornWorker "
                "--keep-alive 65 --workers 4"
            ),
        ],
    },
]


def build_corpus(corpus_dir: Path) -> None:
    """Writes markdown runbooks to disk."""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for spec in RUNBOOK_SPECS:
        runbook_id = str(spec["runbook_id"])
        md_text = _format_runbook(
            runbook_id=runbook_id,
            title=str(spec["title"]),
            error_code=str(spec["error_code"]),
            log_pattern=str(spec["log_pattern"]),
            drivers=str(spec["drivers"]),
            root_cause=str(spec["root_cause"]),
            commands=list(spec["commands"]),  # type: ignore[arg-type]
        )
        (corpus_dir / f"{runbook_id}.md").write_text(md_text, encoding="utf-8")
    print(f"Created {len(RUNBOOK_SPECS)} SRE runbooks in {corpus_dir}.")


if __name__ == "__main__":
    build_corpus(Path("data/runbooks_corpus"))
