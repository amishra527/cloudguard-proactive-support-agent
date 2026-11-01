"""
Cost-Sensitive Threshold Optimizer for Proactive Customer Support.
Translates classification confusion matrices into Net Revenue Retention (NRR) & SLA savings.
"""

from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class BusinessEconomicsConfig:
    """Financial parameters for Google Cloud proactive support intervention."""

    avg_sla_penalty_usd: float = 12500.0
    proactive_mitigation_success_rate: float = 0.80
    proactive_engineer_outreach_cost_usd: float = 350.0


def find_optimal_roi_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, config: BusinessEconomicsConfig
) -> Dict[str, float]:
    """
    Sweeps probability thresholds from 0.05 to 0.95 to find the threshold
    that maximizes Expected Net Financial Savings.
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    best_threshold = 0.50
    max_net_savings = -np.inf
    default_50_savings = 0.0

    value_per_tp = (
        config.avg_sla_penalty_usd * config.proactive_mitigation_success_rate
    ) - config.proactive_engineer_outreach_cost_usd
    cost_per_fp = config.proactive_engineer_outreach_cost_usd

    for thresh in thresholds:
        preds = (y_prob >= thresh).astype(int)
        tp_count = int(np.sum((preds == 1) & (y_true == 1)))
        fp_count = int(np.sum((preds == 1) & (y_true == 0)))
        net_savings = (tp_count * value_per_tp) - (fp_count * cost_per_fp)

        if np.isclose(thresh, 0.50, atol=1e-3):
            default_50_savings = float(net_savings)

        if net_savings > max_net_savings:
            max_net_savings = float(net_savings)
            best_threshold = float(thresh)

    return {
        "optimal_threshold": round(best_threshold, 2),
        "max_net_savings_usd": round(max_net_savings, 2),
        "default_050_savings_usd": round(default_50_savings, 2),
        "incremental_bizops_lift_usd": round(max_net_savings - default_50_savings, 2),
    }
