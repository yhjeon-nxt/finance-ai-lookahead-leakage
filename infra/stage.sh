#!/usr/bin/env bash
# Stage code + prepared price data to S3 so the EC2 instance starts fast (no GitHub auth, no
# re-downloading market data). Run locally before launch_spot.sh.
set -euo pipefail
S3_BUCKET="${S3_BUCKET:-neuroxt-personal}"
S3_PREFIX="${S3_PREFIX:-yhjeon/finance-ai-leakage}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$ROOT"
TAR=$(mktemp -d)/leakage_src.tar.gz
tar czf "$TAR" src requirements.txt
aws s3 cp "$TAR" "s3://$S3_BUCKET/$S3_PREFIX/code/leakage_src.tar.gz"

# Prepared prices (tiny parquet) so EC2 doesn't hit yfinance.
if [ -d data/raw ]; then
  aws s3 sync data/raw "s3://$S3_BUCKET/$S3_PREFIX/data/raw/"
fi
echo "Staged code + data to s3://$S3_BUCKET/$S3_PREFIX/"
