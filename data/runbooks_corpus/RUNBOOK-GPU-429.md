# Runbook ID: RUNBOOK-GPU-429
## Title: Regional NVIDIA A100 GPU Quota Throttling (ResourceExhausted 429)
### Symptom Signature
- Error Code: `GPU_QUOTA_THROTTLING`
- Log Pattern: `ResourceExhausted: Quota 'NVIDIA_A100_GPUS' exceeded in region us-central1`
- Correlated SHAP Drivers: `gpu_throttle_ratio`, `quota_utilization_pct`

### Root Cause Analysis
Autoscaling GPU node pools in a single region hit hard regional project quota limits during peak inference windows.

### Verified Remediation Commands
1. Verified Command 1:
```bash
gcloud compute regions describe us-central1 --format='table(quotas:filter=metric:NVIDIA_A100_GPUS)'
```
2. Verified Command 2:
```bash
gcloud container node-pools update gpu-pool-a100 --enable-autoscaling --max-nodes=24 --region=us-central1
```
