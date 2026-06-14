#!/usr/bin/env bash
# EC2 user-data bootstrap for the look-ahead-leakage experiment.
# Cost discipline: pull tiny prepared data from S3 (not re-download), run once, stream results
# to S3 continuously, and SELF-TERMINATE. Combined with instance-initiated-shutdown-behavior
# =terminate and the resumable decision cache, a spot interruption costs at most one decision.
set -euo pipefail

# cloud-init runs user-data as root WITHOUT $HOME; ollama panics ("$HOME is not defined")
# without it. Set it before anything else.
export HOME="${HOME:-/root}"

S3_BUCKET="${S3_BUCKET:-neuroxt-personal}"
S3_PREFIX="${S3_PREFIX:-yhjeon/finance-ai-leakage}"
RUN_TAG="${RUN_TAG:-ec2}"
WORKDIR=/opt/leakage
LOG=/var/log/leakage_bootstrap.log
exec > >(tee -a "$LOG") 2>&1

echo "=== $(date -u) bootstrap start (tag=$RUN_TAG) ==="

# Always self-terminate on ANY exit (success, failure, or set -e abort) so a crashed bootstrap
# never leaves a GPU instance running and billing. Belt-and-braces with shutdown-behavior.
self_terminate() {
  local rc=$?
  echo "=== $(date -u) exiting rc=$rc — self-terminating ==="
  aws s3 cp "$LOG" "s3://$S3_BUCKET/$S3_PREFIX/$RUN_TAG/bootstrap.log" 2>/dev/null || true
  local TOK IID REG
  TOK=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 60" || true)
  IID=$(curl -s -H "X-aws-ec2-metadata-token: $TOK" http://169.254.169.254/latest/meta-data/instance-id || true)
  REG=$(curl -s -H "X-aws-ec2-metadata-token: $TOK" http://169.254.169.254/latest/meta-data/placement/region || true)
  if [ -n "$IID" ]; then aws ec2 terminate-instances --region "$REG" --instance-ids "$IID" || shutdown -h now
  else shutdown -h now; fi
}
trap self_terminate EXIT

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

# --- models: control + candidate treatments (selection happens after code) --
# qwen3:32b = bigger candidate (fair shot w/ 600s cold-load timeout); qwen3:8b = known-eligible
# fallback (verified: 2/4, denies 2026, sane). qwen2.5:32b dropped — verified older-cutoff (2022/23).
ollama pull llama3.1:8b
ollama pull qwen3:32b
ollama pull qwen3:8b

# --- code + prepared data from S3 ------------------------------------------
mkdir -p "$WORKDIR" && cd "$WORKDIR"
aws s3 cp "s3://$S3_BUCKET/$S3_PREFIX/code/leakage_src.tar.gz" . && tar xzf leakage_src.tar.gz
aws s3 sync "s3://$S3_BUCKET/$S3_PREFIX/data/" "$WORKDIR/data/" || true   # cached prices
# Hard precondition: the prepared parquets MUST be present, else the run would silently fall
# back to live yfinance on EC2 (different/blocked data). Fail loudly (trap self-terminates).
test -f "$WORKDIR/data/raw/prices_2024H2.parquet" || { echo "MISSING prices_2024H2.parquet"; exit 2; }
test -f "$WORKDIR/data/raw/prices_2026JanMay.parquet" || { echo "MISSING prices_2026JanMay.parquet"; exit 2; }
pip3 install -r requirements.txt

# --- run (resumable) + stream results --------------------------------------
sync_results() { aws s3 sync "$WORKDIR/results/" "s3://$S3_BUCKET/$S3_PREFIX/$RUN_TAG/results/" || true; }
( while true; do sync_results; sleep 60; done ) &
RESYNC_PID=$!

# Empirically select the treatment model: highest verified 2024-H2 recall among candidates that
# also deny 2026 knowledge + pass sanity (pick_best fails loudly if none qualify). qwen3:8b is a
# guaranteed-eligible fallback so the bigger models are preferred only when they actually qualify.
export LEAKAGE_CONTROL_MODEL=llama3.1:8b
TREAT=$(PYTHONPATH="$WORKDIR/src" python3 -m leakage.run.model_selection --pick qwen3:32b qwen3:8b 2>>"$LOG" | tail -1)
echo "selected treatment model: '$TREAT'"
[ -n "$TREAT" ] || { echo "treatment selection produced no model — aborting"; exit 3; }
export LEAKAGE_TREATMENT_MODEL="$TREAT"
export LEAKAGE_TREATMENT_CUTOFF="2024+ (auto-selected)"

set +e
PYTHONPATH="$WORKDIR/src" python3 -m leakage.run.main --tag "$RUN_TAG" --no-s3
RC=$?
set -e

kill "$RESYNC_PID" 2>/dev/null || true
sync_results
kill "$STREAM_PID" 2>/dev/null || true
echo "=== $(date -u) bootstrap done rc=$RC ==="
# self_terminate runs automatically via the EXIT trap.
exit "$RC"
