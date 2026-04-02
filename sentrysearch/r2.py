"""Cloudflare R2 (S3-compatible) client for video storage."""

import os
import tempfile
from pathlib import Path

import boto3


class R2Client:
    """S3-compatible client for Cloudflare R2."""

    def __init__(
        self,
        endpoint_url: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        bucket: str | None = None,
    ):
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url or os.environ["R2_ENDPOINT_URL"],
            aws_access_key_id=access_key_id or os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=secret_access_key or os.environ["R2_SECRET_ACCESS_KEY"],
            region_name="auto",
        )
        self._bucket = bucket or os.environ["R2_BUCKET"]

    @property
    def bucket(self) -> str:
        return self._bucket

    def download(self, key: str, local_path: str) -> str:
        """Download an object from R2 to a local file."""
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        self._client.download_file(self._bucket, key, local_path)
        return local_path

    def download_temp(self, key: str) -> str:
        """Download an object to a temporary file. Caller must clean up."""
        suffix = Path(key).suffix or ".mp4"
        fd, path = tempfile.mkstemp(suffix=suffix, prefix="sentrysearch_")
        os.close(fd)
        self._client.download_file(self._bucket, key, path)
        return path

    def upload(self, local_path: str, key: str) -> str:
        """Upload a local file to R2. Returns the key."""
        self._client.upload_file(local_path, self._bucket, key)
        return key

    def presigned_url(self, key: str, expires_in: int = 3600) -> str:
        """Generate a presigned download URL."""
        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self._bucket, "Key": key},
            ExpiresIn=expires_in,
        )

    def list_objects(self, prefix: str = "") -> list[str]:
        """List object keys under a prefix."""
        keys = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys
