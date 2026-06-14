# Infra — cost-optimized EC2 spot run

The full experiment is a few hundred 8B-model inferences, so it runs in well under an hour on a
single GPU. The design minimizes EC2 wall-clock and never re-does work.

## Cost controls
- **Single GPU spot instance** (`g5.xlarge` ≈ $0.40–0.60/hr spot; `g4dn.xlarge` cheapest).
- **Prepared data staged to S3** → no yfinance on EC2, tiny transfer.
- **Resumable decision cache** keyed by `(group, model, window, seed, date)` → a spot kill costs
  at most one decision; re-launch resumes.
- **Self-terminate** on completion + `instance-initiated-shutdown-behavior=terminate`.
- **Live log + result streaming to S3** every 30–60s.
- Expected total cost **< ~$2**.

## Steps
```bash
# 0. (local) prepare price caches if not present
PYTHONPATH=src python3 -m leakage.data.ingest

# 1. stage code + data to S3
bash infra/stage.sh

# 2. launch the spot instance (needs an IAM instance profile with S3 access)
INSTANCE_PROFILE=<role-with-s3> REGION=us-east-1 bash infra/launch_spot.sh

# 3. watch progress
aws s3 cp s3://neuroxt-personal/yhjeon/finance-ai-leakage/ec2/bootstrap.log - | tail

# 4. pull results back when done
aws s3 sync s3://neuroxt-personal/yhjeon/finance-ai-leakage/ec2/results/ results/
```

## Alternative: run locally (zero cloud cost)
The same entrypoint runs on any machine with ollama + a GPU/Apple-Silicon:
```bash
ollama pull llama3.1:8b && ollama pull qwen3:8b
PYTHONPATH=src python3 -m leakage.run.main --tag local
```
