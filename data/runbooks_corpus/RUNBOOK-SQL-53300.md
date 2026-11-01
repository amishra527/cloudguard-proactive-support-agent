# Runbook ID: RUNBOOK-SQL-53300
## Title: Cloud SQL PostgreSQL Connection Pool Exhaustion (SQLSTATE 53300)
### Symptom Signature
- Error Code: `CLOUD_SQL_POOL_EXHAUSTED`
- Log Pattern: `FATAL: remaining connection slots are reserved for non-replication superuser connections`
- Correlated SHAP Drivers: `db_conn_pool_saturation_pct`, `p99_latency_ms`, `api_5xx_error_rate_pct`

### Root Cause Analysis
Microservice instances open unpooled database connections without PgBouncer multiplexing, saturating max_connections during traffic bursts.

### Verified Remediation Commands
1. Verified Command 1:
```bash
gcloud sql instances patch prod-pg-instance --database-flags=max_connections=500
```
2. Verified Command 2:
```bash
export DB_POOL_SIZE=20 DB_MAX_OVERFLOW=10 DB_POOL_RECYCLE_SECONDS=1800
```
