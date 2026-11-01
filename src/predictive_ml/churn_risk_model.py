"""
Predictive SLA Breach & Churn Risk Pipeline with SHAP Explainability.
Outputs structured risk payloads to trigger the Phase 2 LangGraph Diagnostic Agent.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.predictive_ml.cost_sensitive_threshold import (
    BusinessEconomicsConfig,
    find_optimal_roi_threshold,
)

FEATURE_COLUMNS = [
    "quota_utilization_pct",
    "db_conn_pool_saturation_pct",
    "k8s_oom_events_24h",
    "p99_latency_ms",
    "api_5xx_error_rate_pct",
    "gpu_throttle_ratio",
    "support_ticket_velocity_7d",
    "days_since_last_arch_review",
]


def _build_shap_payloads(
    test_df: pd.DataFrame, shap_matrix: np.ndarray, threshold: float
) -> List[Dict[str, Any]]:
    """Extracts top 3 SHAP risk drivers for accounts exceeding the ROI threshold."""
    payloads: List[Dict[str, Any]] = []
    for row_idx, (_, row) in enumerate(test_df.iterrows()):
        if row["predicted_breach_prob"] < threshold:
            continue

        row_shap = shap_matrix[row_idx]
        top_indices = np.argsort(np.abs(row_shap))[::-1][:3]
        drivers = [
            {
                "feature": FEATURE_COLUMNS[i],
                "value": float(row[FEATURE_COLUMNS[i]]),
                "shap_impact": round(float(row_shap[i]), 4),
            }
            for i in top_indices
        ]
        payloads.append({
            "account_id": str(row["account_id"]),
            "account_tier": str(row["account_tier"]),
            "contract_arr_usd": float(row["contract_arr_usd"]),
            "predicted_breach_prob": round(float(row["predicted_breach_prob"]), 4),
            "optimal_roi_threshold_used": threshold,
            "top_shap_risk_drivers": drivers,
        })

    # Sort by highest risk probability descending so top alerts are processed first
    payloads.sort(key=lambda x: x["predicted_breach_prob"], reverse=True)
    return payloads


def train_and_export_agent_triggers(data_path: Path, output_dir: Path) -> None:
    """Trains XGBoost classifier, computes SHAP drivers, and exports agent alert payloads."""
    df = pd.read_csv(data_path)
    splits = train_test_split(
        df[FEATURE_COLUMNS],
        df["sla_breach_48h"].to_numpy(),
        df.index,
        test_size=0.25,
        random_state=42,
        stratify=df["sla_breach_48h"].to_numpy(),
    )
    x_train, x_test, y_train, y_test, _, idx_test = splits

    model = XGBClassifier(
        n_estimators=180,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(x_train, y_train)

    y_prob = model.predict_proba(x_test)[:, 1]
    roi_metrics = find_optimal_roi_threshold(y_test, y_prob, BusinessEconomicsConfig())

    test_df = df.loc[idx_test].copy()
    test_df["predicted_breach_prob"] = y_prob
    shap_matrix = shap.TreeExplainer(model).shap_values(x_test)

    payloads = _build_shap_payloads(test_df, shap_matrix, roi_metrics["optimal_threshold"])

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "agent_trigger_queue.json", "w", encoding="utf-8") as f_out:
        json.dump(payloads[:50], f_out, indent=2)

    metrics_summary = {
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_test, y_prob)), 4),
        **roi_metrics,
        "flagged_high_risk_accounts": len(payloads),
    }
    with open(output_dir / "phase1_model_metrics.json", "w", encoding="utf-8") as f_out:
        json.dump(metrics_summary, f_out, indent=2)

    print(f"Model Trained | ROC-AUC: {metrics_summary['roc_auc']:.4f}")
    print(f"Incremental BizOps Lift: +${roi_metrics['incremental_bizops_lift_usd']:,.2f}")
    print(f"Exported top 50 high-risk alerts to {output_dir / 'agent_trigger_queue.json'}")


if __name__ == "__main__":
    train_and_export_agent_triggers(
        data_path=Path("data/processed/enterprise_telemetry.csv"),
        output_dir=Path("data/processed"),
    )
