# Runbook ID: RUNBOOK-GW-504
## Title: Cloud Load Balancing & API Gateway 5xx Upstream Timeout Spike
### Symptom Signature
- Error Code: `API_GATEWAY_5XX_SPIKE`
- Log Pattern: `UpstreamTimeoutError: Back-end service failed to respond within 5000ms SLA window`
- Correlated SHAP Drivers: `api_5xx_error_rate_pct`, `support_ticket_velocity_7d`, `days_since_last_arch_review`

### Root Cause Analysis
Keep-alive timeout mismatch between Cloud Load Balancer and upstream FastAPI workers causes premature 502/504 connection drops.

### Verified Remediation Commands
1. Verified Command 1:
```bash
gcloud compute backend-services update prod-api-backend --timeout=60s --global
```
2. Verified Command 2:
```bash
gunicorn src.api.main:app -k uvicorn.workers.UvicornWorker --keep-alive 65 --workers 4
```
