"""
Propositional Grounding & Citation Guardrail.
Verifies that agent remediation commands and telemetry claims match authoritative sources.
"""

from typing import Dict, List


def verify_remediation_grounding(
    proposed_commands: List[str],
    retrieved_runbook: Dict[str, object],
    customer_log: Dict[str, object],
    diagnosis_summary: str,
) -> Dict[str, object]:
    """
    Validates that:
    1. Every proposed CLI command exists verbatim in the retrieved runbook.
    2. The diagnosis references the exact error_code from the customer's log.
    """
    allowed_commands = [str(c) for c in retrieved_runbook.get("verified_commands", [])]
    unverified_cmds = [cmd for cmd in proposed_commands if cmd not in allowed_commands]

    error_code = str(customer_log.get("error_code", ""))
    error_code_grounded = error_code in diagnosis_summary

    total_checks = len(proposed_commands) + 1
    passed_checks = (len(proposed_commands) - len(unverified_cmds)) + (
        1 if error_code_grounded else 0
    )
    faithfulness_score = round(passed_checks / max(1, total_checks), 4)

    is_valid = not unverified_cmds and error_code_grounded
    status = "SUPPORTED" if is_valid else "CONTRADICTED"
    return {
        "grounding_status": status,
        "faithfulness_score": faithfulness_score,
        "cited_runbook_id": str(retrieved_runbook.get("runbook_id", "UNKNOWN")),
        "unverified_commands_blocked": unverified_cmds,
        "error_code_verified": error_code_grounded,
    }
