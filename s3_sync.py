# s3_sync.py
import os
from pathlib import Path
import boto3

def download_s3_to_local(bucket: str, key: str, local_path: Path):
    local_path.parent.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, str(local_path))
    return local_path
