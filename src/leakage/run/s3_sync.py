"""Sync result artifacts to S3 (personal scratch bucket per policy).

Uploads continuously during the run so a spot interruption never loses completed work.
No-ops gracefully if boto3 / credentials are unavailable (logged, not fatal).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from leakage.config import S3_BUCKET, S3_PREFIX  # noqa: E402


def upload_dir(local_dir: Path, subprefix: str = "", bucket: str = S3_BUCKET,
               prefix: str = S3_PREFIX) -> int:
    """Upload all files under local_dir to s3://bucket/prefix/subprefix/. Returns count."""
    try:
        import boto3
    except ImportError:
        print("[s3] boto3 not installed; skipping upload")
        return 0
    local_dir = Path(local_dir)
    if not local_dir.exists():
        return 0
    s3 = boto3.client("s3")
    n = 0
    base = "/".join(p for p in [prefix, subprefix] if p)
    for f in local_dir.rglob("*"):
        if f.is_file():
            key = f"{base}/{f.relative_to(local_dir).as_posix()}"
            try:
                s3.upload_file(str(f), bucket, key)
                n += 1
            except Exception as e:  # noqa: BLE001
                print(f"[s3] failed {f}: {e}")
    print(f"[s3] uploaded {n} files to s3://{bucket}/{base}/")
    return n


if __name__ == "__main__":
    from leakage.config import RESULTS_DIR
    upload_dir(RESULTS_DIR, subprefix="results")
