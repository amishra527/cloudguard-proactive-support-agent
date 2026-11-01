"""
Automated Pytest Suite for CloudGuard Predictive ML, Hybrid RAG, Guardrails, and FastAPI.
"""

from fastapi.testclient import TestClient
import numpy as np

from src.agent_engine.hallucination_guard import verify_remediation_grounding
from src.agent_engine.milvus_hybrid_rag import HybridRunbookRetriever
from src.api.main import app
from src.evaluation.agent_benchmark import EXPECTED_RUNBOOK_MAP
from src.predictive_ml.cost_sensitive_threshold import (
    BusinessEconomicsConfig,
    find_optimal_roi_threshold,
)

CLIENT = TestClient(app)


def test_cost_sensitive_threshold_beats_naive() -> None:
    """Verifies that cost-sensitive ROI optimization outperforms or equals 0.50 default."""
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    y_prob = np.array([0.02, 0.08, 0.22, 0.78, 0.12, 0.35, 0.04, 0.09, 0.65, 0.01])
    res = find_optimal_roi_threshold(y_true, y_prob, BusinessEconomicsConfig())
    assert res["max_net_savings_usd"] >= res["default_050_savings_usd"]
    assert 0.05 <= res["optimal_threshold"] <= 0.95


def test_hybrid_rag_retrieves_exact_runbook() -> None:
    """Verifies that K8S_OOM_KILLED queries retrieve RUNBOOK-K8S-137 at rank 1."""
    retriever = HybridRunbookRetriever()
    docs = retriever.retrieve("K8S_OOM_KILLED Container memory limit exceeded exit code 137")
    assert docs[0]["runbook_id"] == "RUNBOOK-K8S-137"
    assert float(docs[0]["retrieval_confidence"]) > 0.40


def test_propositional_guard_blocks_rogue_command() -> None:
    """Verifies that unverified CLI commands trigger a CONTRADICTED guardrail status."""
    runbook = {"runbook_id": "RUNBOOK-K8S-137", "verified_commands": ["kubectl get pods"]}
    log_entry = {"error_code": "K8S_OOM_KILLED"}
    report = verify_remediation_grounding(
        proposed_commands=["kubectl get pods", "rm -rf /"],
        retrieved_runbook=runbook,
        customer_log=log_entry,
        diagnosis_summary="Detected K8S_OOM_KILLED on worker.",
    )
    assert report["grounding_status"] == "CONTRADICTED"
    assert "rm -rf /" in list(report["unverified_commands_blocked"])  # type: ignore[call-overload]


def test_fastapi_diagnose_endpoint() -> None:
    """Tests end-to-end FastAPI /agent/diagnose contract and grounding verification."""
    payload = {
        "account_id": "GCP-ENT-10040",
        "account_tier": "Strategic",
        "contract_arr_usd": 342300.0,
        "predicted_breach_prob": 0.7373,
        "top_shap_risk_drivers": [
            {"feature": "k8s_oom_events_24h", "value": 11.0, "shap_impact": 1.4992}
        ],
    }
    response = CLIENT.post("/agent/diagnose", json=payload)
    assert response.status_code == 200
    data = response.json()
    expected_runbook = EXPECTED_RUNBOOK_MAP[data["detected_error_code"]]
    assert data["cited_runbook_id"] == expected_runbook
    assert data["guardrail_verification"]["grounding_status"] == "SUPPORTED"
