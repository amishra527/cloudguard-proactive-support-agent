"""
Synthetic Enterprise Cloud Telemetry & Diagnostic Log Generator.
Simulates 5,000 enterprise accounts with telemetry metrics, ARR, and failure logs.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import json
import numpy as np
import pandas as pd


def _build_diagnostic_log_entry(row: pd.Series) -> Dict[str, object]:
    """Determines the primary symptom and raw log snippet for a given account row."""
    if row["db_conn_pool_saturation_pct"] > 80.0:
        symptom = "CLOUD_SQL_POOL_EXHAUSTED"
        msg = (
            "FATAL: remaining connection slots are reserved for "
            "non-replication superuser connections."
        )
    elif row["gpu_throttle_ratio"] > 0.30:
        symptom = "GPU_QUOTA_THROTTLING"
        msg = "ResourceExhausted: Quota 'NVIDIA_A100_GPUS' exceeded in region us-central1."
    elif row["k8s_oom_events_24h"] >= 5:
        symptom = "K8S_OOM_KILLED"
        msg = "Container memory limit exceeded (exit code 137) on pod inference-worker."
    else:
        symptom = "API_GATEWAY_5XX_SPIKE"
        msg = "UpstreamTimeoutError: Back-end service failed to respond within 5000ms SLA window."

    return {
        "account_id": str(row["account_id"]),
        "error_code": symptom,
        "p99_latency_ms": float(row["p99_latency_ms"]),
        "raw_log_snippet": msg,
    }


def generate_enterprise_telemetry(
    num_accounts: int = 5000, random_seed: int = 42
) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    """Generates synthetic cloud telemetry dataset and corresponding error logs."""
    rng = np.random.default_rng(random_seed)
    tiers = rng.choice(
        ["Enterprise", "Strategic", "Mid-Market"],
        size=num_accounts,
        p=[0.35, 0.15, 0.50],
    )
    base_arr = np.where(
        tiers == "Strategic", 500000, np.where(tiers == "Enterprise", 180000, 60000)
    )

    quota_pct = np.clip(rng.normal(68, 18, num_accounts), 10.0, 100.0)
    db_pool_pct = np.clip(rng.normal(55, 22, num_accounts), 5.0, 100.0)
    oom_events = rng.poisson(lam=np.exp((quota_pct - 60) / 20), size=num_accounts)
    api_5xx_pct = np.clip(
        0.1
        + 0.04 * oom_events
        + 0.03 * np.maximum(0, db_pool_pct - 75)
        + rng.exponential(0.3, num_accounts),
        0.0,
        25.0,
    )
    gpu_throttle = np.clip(
        rng.beta(a=1.5, b=8.0, size=num_accounts) * (quota_pct / 80), 0.0, 1.0
    )
    tickets_7d = rng.poisson(lam=1.2 + 0.35 * api_5xx_pct, size=num_accounts)
    arch_days = rng.integers(7, 365, size=num_accounts)

    logit = (
        -6.2
        + 0.045 * np.maximum(0, quota_pct - 75)
        + 0.055 * np.maximum(0, db_pool_pct - 70)
        + 0.38 * oom_events
        + 0.28 * api_5xx_pct
        + 1.8 * gpu_throttle
        + 0.22 * tickets_7d
        + 0.003 * arch_days
    )
    df = pd.DataFrame({
        "account_id": [f"GCP-ENT-{10000 + i}" for i in range(num_accounts)],
        "account_tier": tiers,
        "contract_arr_usd": np.round(
            base_arr * rng.lognormal(mean=0.0, sigma=0.4, size=num_accounts), -2
        ),
        "quota_utilization_pct": np.round(quota_pct, 2),
        "db_conn_pool_saturation_pct": np.round(db_pool_pct, 2),
        "k8s_oom_events_24h": oom_events,
        "p99_latency_ms": np.round(
            np.clip(
                120 + (db_pool_pct * 3.5) + (oom_events * 45) + rng.normal(0, 35, num_accounts),
                40.0,
                5000.0,
            ),
            1,
        ),
        "api_5xx_error_rate_pct": np.round(api_5xx_pct, 2),
        "gpu_throttle_ratio": np.round(gpu_throttle, 3),
        "support_ticket_velocity_7d": tickets_7d,
        "days_since_last_arch_review": arch_days,
        "sla_breach_48h": rng.binomial(1, 1 / (1 + np.exp(-logit))),
    })

    # Generate diagnostic logs for all accounts showing elevated telemetry signals
    elevated_mask = (
        (df["sla_breach_48h"] == 1)
        | (df["quota_utilization_pct"] > 75)
        | (df["db_conn_pool_saturation_pct"] > 75)
        | (df["k8s_oom_events_24h"] >= 4)
        | (df["gpu_throttle_ratio"] > 0.20)
        | (df["support_ticket_velocity_7d"] >= 3)
    )
    logs = [_build_diagnostic_log_entry(row) for _, row in df[elevated_mask].iterrows()]
    return df, logs


if __name__ == "__main__":
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    telemetry_df, diag_logs = generate_enterprise_telemetry()
    telemetry_df.to_csv(out_dir / "enterprise_telemetry.csv", index=False)

    with open(out_dir / "diagnostic_logs.json", "w", encoding="utf-8") as f_out:
        json.dump(diag_logs, f_out, indent=2)

    breach_pct = telemetry_df["sla_breach_48h"].mean() * 100
    print(f"Generated {len(telemetry_df)} accounts (SLA Breach Rate: {breach_pct:.1f}%).")
    print(f"Saved {len(diag_logs)} diagnostic log records to {out_dir}.")
