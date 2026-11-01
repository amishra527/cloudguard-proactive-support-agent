"""
Pydantic Schemas for CloudGuard Proactive Support API.
Enforces strict JSON contracts for telemetry triggers and grounded agent advisories.
"""

from typing import Dict, List
from pydantic import BaseModel, Field


class ShapDriverSchema(BaseModel):
    """Represents a single SHAP feature attribution for an account alert."""

    feature: str
    value: float
    shap_impact: float


class AccountTriggerRequest(BaseModel):
    """Input payload for triggering a proactive diagnostic agent run."""

    account_id: str = Field(..., examples=["GCP-ENT-10219"])
    account_tier: str = Field(..., examples=["Strategic"])
    contract_arr_usd: float = Field(..., gt=0)
    predicted_breach_prob: float = Field(..., ge=0.0, le=1.0)
    top_shap_risk_drivers: List[ShapDriverSchema]


class ProactiveAdvisoryResponse(BaseModel):
    """Output contract returned by the autonomous diagnostic and remediation agent."""

    account_id: str
    account_tier: str
    contract_arr_usd: float
    predicted_breach_prob: float
    detected_error_code: str
    diagnosis_summary: str
    recommended_cli_commands: List[str]
    cited_runbook_id: str
    retrieval_confidence: float
    guardrail_verification: Dict[str, object]
    agent_trajectory: List[str]
    execution_latency_ms: float
