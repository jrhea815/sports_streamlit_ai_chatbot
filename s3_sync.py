# s3_sync.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import boto3


@dataclass(frozen=True)
class S3ObjectRef:
    bucket: str
    key: str


def ensure_latest_local_csv(
    s3_ref: S3ObjectRef,
    local_path: Path,
    *,
    force: bool = False,
) -> Path:
    """
    Ensures local_path exists and is the latest version from S3.

    - If local file doesn't exist -> downloads
    - If local exists -> compares S3 LastModified vs local mtime (UTC-ish) and downloads if S3 is newer
    - If force=True -> always downloads
    """
    local_path.parent.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3")  # On EC2, uses instance role automatically

    # If we don't have the file yet, download it
    if force or not local_path.exists():
        s3.download_file(s3_ref.bucket, s3_ref.key, str(local_path))
        return local_path

    # Compare remote LastModified to local mtime
    head = s3.head_object(Bucket=s3_ref.bucket, Key=s3_ref.key)
    remote_last_modified = head["LastModified"]  # tz-aware datetime

    local_mtime = local_path.stat().st_mtime  # seconds since epoch
    remote_ts = remote_last_modified.timestamp()

    if remote_ts > local_mtime:
        s3.download_file(s3_ref.bucket, s3_ref.key, str(local_path))

    return local_path

