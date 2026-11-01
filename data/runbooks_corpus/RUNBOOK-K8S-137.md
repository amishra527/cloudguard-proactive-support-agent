# Runbook ID: RUNBOOK-K8S-137
## Title: GKE Container OOMKilled (Exit Code 137) on Inference Workers
### Symptom Signature
- Error Code: `K8S_OOM_KILLED`
- Log Pattern: `Container memory limit exceeded (exit code 137) on pod inference-worker`
- Correlated SHAP Drivers: `k8s_oom_events_24h`, `quota_utilization_pct`, `p99_latency_ms`

### Root Cause Analysis
Batch payload spikes during concurrent inference requests exceed pod cgroup memory ceiling, causing Linux kernel OOM killer termination.

### Verified Remediation Commands
1. Verified Command 1:
```bash
kubectl set resources deployment inference-worker -c=inference-worker --limits=memory=16Gi --requests=memory=12Gi
```
2. Verified Command 2:
```bash
kubectl autoscale deployment inference-worker --cpu-percent=75 --min=3 --max=15
```
