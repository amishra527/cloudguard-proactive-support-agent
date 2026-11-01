"""
FastAPI Production Service for CloudGuard Proactive Support Agent.
Exposes REST endpoints for real-time incident diagnosis and BizOps ROI metrics.
"""

import json
from pathlib import Path
from typing import Dict
from fastapi import FastAPI

from src.agent_engine.graph import ProactiveSupportAgent
from src.api.schemas import AccountTriggerRequest, ProactiveAdvisoryResponse

app = FastAPI(
    title="CloudGuard Proactive Support & BizOps API",
    version="1.0.0",
    description="Predictive SLA breach detection and grounded multi-step remediation agent.",
)

AGENT_INSTANCE = ProactiveSupportAgent()


@app.get("/health")
def health_check() -> Dict[str, object]:
    """Liveness and readiness probe for Kubernetes orchestration."""
    return {
        "status": "healthy",
        "indexed_accounts": AGENT_INSTANCE.telemetry_tool.count_indexed_accounts(),
        "loaded_runbooks": len(AGENT_INSTANCE.rag_retriever.documents),
    }


@app.get("/metrics/bizops-summary")
def get_bizops_summary() -> Dict[str, object]:
    """Returns combined Phase 1 predictive ML ROI and Phase 3 agent evaluation metrics."""
    p1_path = Path("data/processed/phase1_model_metrics.json")
    p3_path = Path("data/processed/phase3_agent_benchmark.json")

    p1_data = json.loads(p1_path.read_text(encoding="utf-8")) if p1_path.exists() else {}
    p3_data = json.loads(p3_path.read_text(encoding="utf-8")) if p3_path.exists() else {}
    return {"predictive_ml_roi": p1_data, "agent_evaluation_benchmark": p3_data}


@app.post("/agent/diagnose", response_model=ProactiveAdvisoryResponse)
def diagnose_account(request: AccountTriggerRequest) -> Dict[str, object]:
    """Executes the 3-node proactive diagnostic and grounded remediation workflow."""
    payload = request.model_dump()
    return AGENT_INSTANCE.run_single_account(payload)
