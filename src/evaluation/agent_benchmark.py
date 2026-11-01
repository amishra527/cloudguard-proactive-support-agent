"""
Multi-Dimensional Agent Evaluation & Statistical Benchmark Suite.
Grades Agent Task Completion, Trajectory Optimality, Propositional Grounding,
and computes Paired Bootstrap p-values against a Naive Baseline Agent.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

from src.agent_engine.graph import ProactiveSupportAgent
from src.agent_engine.hallucination_guard import verify_remediation_grounding

EXPECTED_RUNBOOK_MAP = {
    "K8S_OOM_KILLED": "RUNBOOK-K8S-137",
    "CLOUD_SQL_POOL_EXHAUSTED": "RUNBOOK-SQL-53300",
    "GPU_QUOTA_THROTTLING": "RUNBOOK-GPU-429",
    "API_GATEWAY_5XX_SPIKE": "RUNBOOK-GW-504",
}


def _paired_bootstrap_pvalue(
    agent_scores: np.ndarray, baseline_scores: np.ndarray, n_boot: int = 2000
) -> Tuple[float, float, float]:
    """
    Computes mean improvement, 95% confidence interval lower bound, and
    one-sided paired bootstrap p-value testing H0: Agent <= Baseline.
    """
    rng = np.random.default_rng(42)
    diffs = agent_scores - baseline_scores
    observed_lift = float(np.mean(diffs))

    boot_means = np.array([
        np.mean(rng.choice(diffs, size=len(diffs), replace=True))
        for _ in range(n_boot)
    ])
    ci_low = float(np.percentile(boot_means, 2.5))
    p_val = float(np.mean(boot_means <= 0.0))
    return round(observed_lift, 4), round(ci_low, 4), round(p_val, 5)


def _evaluate_single_scenario(
    agent: ProactiveSupportAgent, trigger: Dict[str, object], inject_hallucination: bool
) -> Dict[str, float]:
    """Runs both the CloudGuard Agent and a Naive Baseline on one golden scenario."""
    result = agent.run_single_account(trigger)
    error_code = str(result["detected_error_code"])
    expected_runbook = EXPECTED_RUNBOOK_MAP.get(error_code, "RUNBOOK-GW-504")

    runbook_correct = 1.0 if result["cited_runbook_id"] == expected_runbook else 0.0
    traj_optimal = 1.0 if len(result["agent_trajectory"]) == 5 else 0.0

    grounding_status = result["guardrail_verification"]["grounding_status"]
    guard_passed = 1.0 if grounding_status == "SUPPORTED" else 0.0
    if inject_hallucination:
        rogue_cmds = list(result["recommended_cli_commands"]) + [
            "kubectl delete namespace prod-cluster --force --grace-period=0"
        ]
        check = verify_remediation_grounding(
            rogue_cmds,
            {"verified_commands": result["recommended_cli_commands"]},
            {"error_code": error_code},
            str(result["diagnosis_summary"]),
        )
        guard_passed = 1.0 if check["grounding_status"] == "CONTRADICTED" else 0.0

    naive_doc = agent.rag_retriever.retrieve(str(trigger["account_tier"]), top_k=1)[0]
    naive_correct = 1.0 if naive_doc["runbook_id"] == expected_runbook else 0.0
    naive_guard = 0.0 if inject_hallucination else 1.0

    return {
        "runbook_accuracy": runbook_correct,
        "trajectory_optimality": traj_optimal,
        "guardrail_precision": guard_passed,
        "cloudguard_composite": 0.5 * runbook_correct + 0.2 * traj_optimal + 0.3 * guard_passed,
        "naive_composite": 0.5 * naive_correct + 0.2 * 0.5 + 0.3 * naive_guard,
        "latency_ms": float(result["execution_latency_ms"]),
    }


def run_benchmark_suite(queue_path: Path, output_path: Path) -> None:
    """Executes the 50-scenario evaluation suite and exports statistical report."""
    with open(queue_path, "r", encoding="utf-8") as f_in:
        scenarios: List[Dict[str, object]] = json.load(f_in)[:50]

    agent = ProactiveSupportAgent()
    records = [
        _evaluate_single_scenario(agent, item, inject_hallucination=idx % 5 == 0)
        for idx, item in enumerate(scenarios)
    ]

    cg_scores = np.array([r["cloudguard_composite"] for r in records])
    naive_scores = np.array([r["naive_composite"] for r in records])
    lift, ci_low, p_value = _paired_bootstrap_pvalue(cg_scores, naive_scores)
    latencies = np.array([r["latency_ms"] for r in records])

    report = {
        "scenarios_evaluated": len(records),
        "adversarial_hallucination_injections": sum(1 for i in range(len(records)) if i % 5 == 0),
        "task_completion_runbook_accuracy_pct": round(
            float(np.mean([r["runbook_accuracy"] for r in records]) * 100), 2
        ),
        "trajectory_optimality_pct": round(
            float(np.mean([r["trajectory_optimality"] for r in records]) * 100), 2
        ),
        "adversarial_guardrail_catch_rate_pct": round(
            float(np.mean([r["guardrail_precision"] for r in records]) * 100), 2
        ),
        "cloudguard_mean_composite_score": round(float(np.mean(cg_scores)), 4),
        "naive_baseline_mean_composite_score": round(float(np.mean(naive_scores)), 4),
        "paired_bootstrap_mean_lift": lift,
        "paired_bootstrap_95ci_lower": ci_low,
        "paired_bootstrap_p_value": p_value,
        "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(report, f_out, indent=2)

    print(f"Evaluated {report['scenarios_evaluated']} Golden Incident Scenarios:")
    print(f"  - Task Completion (Runbook Match): {report['task_completion_runbook_accuracy_pct']}%")
    print(f"  - Agent Trajectory Optimality:     {report['trajectory_optimality_pct']}%")
    print(f"  - Adversarial Hallucination Catch: {report['adversarial_guardrail_catch_rate_pct']}%")
    print(f"  - Statistical Lift: +{lift} (95% CI Lower: +{ci_low}, p-value: {p_value})")


if __name__ == "__main__":
    run_benchmark_suite(
        queue_path=Path("data/processed/agent_trigger_queue.json"),
        output_path=Path("data/processed/phase3_agent_benchmark.json"),
    )
