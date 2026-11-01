"""
Executive BizOps ROI Simulator & Live Autonomous Agent Operations Dashboard.
Run with: streamlit run dashboard/bizops_roi_simulator.py
"""

import json
from pathlib import Path
from typing import Dict, List
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="CloudGuard BizOps & Agent Studio",
    layout="wide",
)


def load_json_artifact(path_str: str) -> object:
    """Loads a JSON artifact from data/processed."""
    path = Path(path_str)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _render_agent_ops_tab(interventions: List[Dict[str, object]]) -> None:
    """Renders Tab 1: Live Proactive Agent Ops."""
    if not interventions:
        st.warning("No proactive intervention records found.")
        return

    labels = [
        f"{item['account_id']} ({item['account_tier']} | "
        f"${float(item['contract_arr_usd']):,.0f} ARR | "
        f"Risk: {float(item['predicted_breach_prob']):.1%})"
        for item in interventions
    ]
    selected_label = st.selectbox("Select High-Risk Enterprise Account:", labels)
    record = interventions[labels.index(selected_label)]

    left_col, right_col = st.columns([3, 2])
    with left_col:
        st.subheader("Autonomous Diagnosis & Cited Remediation")
        st.info(str(record["diagnosis_summary"]))
        st.markdown(f"**Cited SRE Runbook:** `{record['cited_runbook_id']}`")
        for cmd in list(record["recommended_cli_commands"]):  # type: ignore[call-overload]
            st.code(str(cmd), language="bash")

    with right_col:
        st.subheader("Propositional Guardrail & Trajectory")
        st.json(record["guardrail_verification"])
        st.markdown("**Executed Agent Tool Trajectory:**")
        traj = [str(step) for step in list(record["agent_trajectory"])]  # type: ignore[call-overload]
        st.write(" -> ".join(traj))


def _render_roi_simulator_tab(p1_metrics: Dict[str, float]) -> None:
    """Renders Tab 2: Interactive Customer Support Economics Simulator."""
    st.subheader("Interactive Customer Support Economics Simulator")
    s_col1, s_col2, s_col3 = st.columns(3)
    sla_penalty = s_col1.slider("Avg Enterprise SLA Penalty ($)", 5000, 50000, 12500, 500)
    mitigation_rate = s_col2.slider("Proactive Prevention Rate (%)", 40, 95, 80, 5) / 100.0
    outreach_cost = s_col3.slider("Proactive Support Cost / Alert ($)", 50, 1000, 350, 50)

    flagged_count = int(p1_metrics.get("flagged_high_risk_accounts", 250))
    est_true_positives = int(flagged_count * 0.28)
    gross_saved = est_true_positives * sla_penalty * mitigation_rate
    total_outreach_spend = flagged_count * outreach_cost
    net_annual_roi = (gross_saved - total_outreach_spend) * 4

    m1, m2, m3 = st.columns(3)
    m1.metric("Quarterly SLA Penalties Avoided", f"${gross_saved:,.0f}")
    m2.metric("Quarterly Proactive Ops Spend", f"${total_outreach_spend:,.0f}")
    m3.metric("Projected Annualized Net Savings", f"${net_annual_roi:,.0f}")

    scenarios = [
        "Reactive Support (Baseline)",
        "Naive 0.50 Threshold",
        "CloudGuard Cost-Sensitive AI",
    ]
    sim_df = pd.DataFrame({
        "Scenario": scenarios,
        "Annualized Net Value ($)": [
            0.0,
            float(p1_metrics.get("default_050_savings_usd", 0.0)) * 4,
            float(net_annual_roi),
        ],
    })
    st.bar_chart(sim_df.set_index("Scenario"))


def render_dashboard() -> None:
    """Renders the two-tab Executive BizOps Simulator and Live Agent Ops UI."""
    st.title("CloudGuard: Proactive Support Autonomous Agent & BizOps Studio")
    st.caption(
        "Google Cloud Senior Business Data Scientist Portfolio Project | "
        "XGBoost + SHAP Risk Scoring | Hybrid Runbook RAG | Propositional Citation Guardrails"
    )

    p1_metrics = load_json_artifact("data/processed/phase1_model_metrics.json")
    p3_metrics = load_json_artifact("data/processed/phase3_agent_benchmark.json")
    interventions = load_json_artifact("data/processed/proactive_interventions.json")

    if not isinstance(p1_metrics, dict) or not isinstance(p3_metrics, dict):
        st.error("Please run Phase 1, 2, and 3 pipelines first.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Predictive Model ROC-AUC", f"{p1_metrics.get('roc_auc', 0):.4f}")
    col2.metric(
        "Optimal ROI Threshold Lift",
        f"${p1_metrics.get('incremental_bizops_lift_usd', 0):,.0f}",
        delta="vs 0.50 naive threshold",
    )
    col3.metric(
        "Agent Task Completion",
        f"{p3_metrics.get('task_completion_runbook_accuracy_pct', 0)}%",
        delta=f"p-val: {p3_metrics.get('paired_bootstrap_p_value', 0)}",
    )
    col4.metric(
        "Hallucination Guard Catch Rate",
        f"{p3_metrics.get('adversarial_guardrail_catch_rate_pct', 0)}%",
        delta=f"p95: {p3_metrics.get('latency_p95_ms', 0)} ms",
    )

    tab1, tab2 = st.tabs([
        "1. Live Proactive Agent Ops (SHAP + RAG + Guardrail)",
        "2. Executive BizOps ROI Simulator",
    ])
    with tab1:
        if isinstance(interventions, list):
            _render_agent_ops_tab(interventions)
    with tab2:
        _render_roi_simulator_tab(p1_metrics)


if __name__ == "__main__":
    render_dashboard()
