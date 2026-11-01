"""
Agent Tool: Queries customer diagnostic logs and telemetry records by account_id.
"""

import json
from pathlib import Path
from typing import Dict, Optional


class TelemetryDiagnosticTool:
    """Provides structured log lookup for the autonomous diagnostic agent."""

    def __init__(
        self, logs_path: Path = Path("data/processed/diagnostic_logs.json")
    ) -> None:
        with open(logs_path, "r", encoding="utf-8") as f_in:
            records = json.load(f_in)
        self._log_index: Dict[str, Dict[str, object]] = {
            str(item["account_id"]): item for item in records
        }

    def fetch_latest_incident_log(
        self, account_id: str
    ) -> Optional[Dict[str, object]]:
        """Returns the most recent structured diagnostic log for an account."""
        return self._log_index.get(account_id)

    def count_indexed_accounts(self) -> int:
        """Returns total number of accounts with active diagnostic logs."""
        return len(self._log_index)
