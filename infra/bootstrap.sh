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
# Wait up to 5 min for the dpkg lock: cloud-init runs its own apt at boot and would otherwise
# race us ("Could not get lock /var/lib/dpkg/lock-frontend").
export DEBIAN_FRONTEND=noninteractive
APT="apt-get -o DPkg::Lock::Timeout=300 -y"
$APT update
$APT install python3-pip awscli curl
curl -fsSL https://ollama.com/install.sh | sh
nohup ollama serve >/var/log/ollama.log 2>&1 &
for i in $(seq 1 30); do curl -s http://localhost:11434/api/version && break; sleep 2; done

# --- models: control + treatment(s) (env-driven) -----------------------------
# LEAKAGE_FORCE_TREATMENT (if set) pins the treatment and skips auto-selection (used for the
# Gemma-3 independent-family replication). Otherwise pull the candidate set and gate-select.
CONTROL_M="${LEAKAGE_CONTROL_MODEL:-llama3.1:8b}"
FORCE_TREAT="${LEAKAGE_FORCE_TREATMENT:-}"
CANDIDATES="${LEAKAGE_TREAT_CANDIDATES:-qwen3:32b qwen3:8b}"
CUSTOM_MODULE="${LEAKAGE_RUN_MODULE:-}"      # e.g. leakage.run.per_model_windows
PULL_MODELS="${LEAKAGE_PULL_MODELS:-}"       # explicit model list for custom-module runs
if [ -n "$CUSTOM_MODULE" ]; then
  for m in $PULL_MODELS; do ollama pull "$m"; done
elif [ -n "$FORCE_TREAT" ]; then
  ollama pull "$CONTROL_M"; ollama pull "$FORCE_TREAT"
else
  ollama pull "$CONTROL_M"; for m in $CANDIDATES; do ollama pull "$m"; done
fi

# --- code + prepared data from S3 ------------------------------------------
mkdir -p "$WORKDIR" && cd "$WORKDIR"
aws s3 cp "s3://$S3_BUCKET/$S3_PREFIX/code/leakage_src.tar.gz" . && tar xzf leakage_src.tar.gz
aws s3 sync "s3://$S3_BUCKET/$S3_PREFIX/data/" "$WORKDIR/data/" || true   # cached prices
# Hard precondition: the prepared parquets MUST be present, else the run would silently fall
# back to live yfinance on EC2 (different/blocked data). Fail loudly (trap self-terminates).
test -f "$WORKDIR/data/raw/prices_2024H2.parquet" || { echo "MISSING prices_2024H2.parquet"; exit 2; }
test -f "$WORKDIR/data/raw/prices_2026JanMay.parquet" || { echo "MISSING prices_2026JanMay.parquet"; exit 2; }
# SPOT RESUME: restore any prior progress (per-day decision cache) from S3, so a re-launched
# instance after a spot interruption continues instead of recomputing from scratch. The decision
# cache is keyed by (group,client,seed,date) and only completed days are cached, so this is safe.
aws s3 sync "s3://$S3_BUCKET/$S3_PREFIX/$RUN_TAG/results/" "$WORKDIR/results/" || true
echo "restored $(find "$WORKDIR/results/decisions" -name '*.jsonl' 2>/dev/null | wc -l) cached decision files"
pip3 install -r requirements.txt

# --- run (resumable) + stream results --------------------------------------
sync_results() { aws s3 sync "$WORKDIR/results/" "s3://$S3_BUCKET/$S3_PREFIX/$RUN_TAG/results/" || true; }
( while true; do sync_results; sleep 60; done ) &
RESYNC_PID=$!

export LEAKAGE_CONTROL_MODEL="$CONTROL_M"
if [ -n "$CUSTOM_MODULE" ]; then
  # Custom experiment module (e.g. per-model in-vs-out); it manages its own models/windows.
  echo "running custom module: $CUSTOM_MODULE"
  set +e
  PYTHONPATH="$WORKDIR/src" python3 -m "$CUSTOM_MODULE"
  RC=$?
  set -e
else
  # Treatment: forced (replication) or gate-selected (highest verified 2024-H2 recall).
  if [ -n "$FORCE_TREAT" ]; then
    TREAT="$FORCE_TREAT"; echo "forced treatment model: '$TREAT'"
  else
    TREAT=$(PYTHONPATH="$WORKDIR/src" python3 -m leakage.run.model_selection --pick $CANDIDATES 2>>"$LOG" | tail -1)
    echo "selected treatment model: '$TREAT'"
  fi
  [ -n "$TREAT" ] || { echo "no treatment model — aborting"; exit 3; }
  export LEAKAGE_TREATMENT_MODEL="$TREAT"
  export LEAKAGE_TREATMENT_CUTOFF="2024+ (${FORCE_TREAT:+forced }auto)"
  set +e
  PYTHONPATH="$WORKDIR/src" python3 -m leakage.run.main --tag "$RUN_TAG" --no-s3
  RC=$?
  set -e
fi

kill "$RESYNC_PID" 2>/dev/null || true
sync_results
kill "$STREAM_PID" 2>/dev/null || true
echo "=== $(date -u) bootstrap done rc=$RC ==="
# self_terminate runs automatically via the EXIT trap.
exit "$RC"
