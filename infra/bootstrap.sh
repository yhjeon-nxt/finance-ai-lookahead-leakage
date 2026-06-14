#!/usr/bin/env bash
# EC2 user-data bootstrap for the look-ahead-leakage experiment.
# Cost discipline: pull tiny prepared data from S3 (not re-download), run once, stream results
# to S3 continuously, and SELF-TERMINATE. Combined with instance-initiated-shutdown-behavior
# =terminate and the resumable decision cache, a spot interruption costs at most one decision.
set -euo pipefail

S3_BUCKET="${S3_BUCKET:-neuroxt-personal}"
S3_PREFIX="${S3_PREFIX:-yhjeon/finance-ai-leakage}"
RUN_TAG="${RUN_TAG:-ec2}"
WORKDIR=/opt/leakage
LOG=/var/log/leakage_bootstrap.log
exec > >(tee -a "$LOG") 2>&1

echo "=== $(date -u) bootstrap start (tag=$RUN_TAG) ==="

stream_logs() {  # background: ship the log to S3 every 30s so we see progress live
  while true; do
    aws s3 cp "$LOG" "s3://$S3_BUCKET/$S3_PREFIX/$RUN_TAG/bootstrap.log" >/dev/null 2>&1 || true
    sleep 30
  done
}
stream_logs &
STREAM_PID=$!

# --- deps -------------------------------------------------------------------
export DEBIAN_FRONTEND=noninteractive
apt-get update -y && apt-get install -y python3-pip awscli curl
curl -fsSL https://ollama.com/install.sh | sh
nohup ollama serve >/var/log/ollama.log 2>&1 &
for i in $(seq 1 30); do curl -s http://localhost:11434/api/version && break; sleep 2; done

# --- models (the two real groups) ------------------------------------------
ollama pull llama3.1:8b
ollama pull qwen3:8b

# --- code + prepared data from S3 ------------------------------------------
mkdir -p "$WORKDIR" && cd "$WORKDIR"
aws s3 cp "s3://$S3_BUCKET/$S3_PREFIX/code/leakage_src.tar.gz" . && tar xzf leakage_src.tar.gz
aws s3 sync "s3://$S3_BUCKET/$S3_PREFIX/data/" "$WORKDIR/data/" || true   # cached prices
pip3 install -r requirements.txt

# --- run (resumable) + stream results --------------------------------------
sync_results() { aws s3 sync "$WORKDIR/results/" "s3://$S3_BUCKET/$S3_PREFIX/$RUN_TAG/results/" || true; }
( while true; do sync_results; sleep 60; done ) &
RESYNC_PID=$!

set +e
PYTHONPATH="$WORKDIR/src" python3 -m leakage.run.main --tag "$RUN_TAG" --no-s3
RC=$?
set -e

kill "$RESYNC_PID" 2>/dev/null || true
sync_results
kill "$STREAM_PID" 2>/dev/null || true
aws s3 cp "$LOG" "s3://$S3_BUCKET/$S3_PREFIX/$RUN_TAG/bootstrap.log" || true

echo "=== $(date -u) bootstrap done rc=$RC — self-terminating ==="
# Self-terminate (defense in depth alongside shutdown-behavior=terminate).
TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 60" || true)
IID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id || true)
REGION=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/placement/region || true)
[ -n "$IID" ] && aws ec2 terminate-instances --region "$REGION" --instance-ids "$IID" || shutdown -h now
