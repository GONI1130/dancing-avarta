"""프레임 Blob 저장소. 멀티레플리카 대응의 핵심.

- FileBlobStore: 기본. tmp/job_{id}.{kind}.json
- S3BlobStore: S3_* 환경이 다 채워져 있을 때 사용 (boto3 lazy import).
  로컬에도 캐시 파일을 남겨 읽기 속도를 유지한다.

kind: frames(원본) | rig(회전값) | bvh 텍스트는 별도 캐시.
S3 키: {prefix}{job_id}.{kind}.json
"""
import json
import logging
import os

log = logging.getLogger(__name__)


class FileBlobStore:
    _EXT = {"genvideo": ".mp4", "control": ".mp4", "audio": ".m4a"}
    _MIME = {"genvideo": "video/mp4", "control": "video/mp4", "audio": "audio/mp4"}
    _BINARY = ("genvideo", "control", "audio")

    def __init__(self, tmp_dir: str) -> None:
        self.tmp_dir = tmp_dir
        os.makedirs(tmp_dir, exist_ok=True)

    def _path(self, job_id: str, kind: str) -> str:
        return os.path.join(self.tmp_dir, f"job_{job_id}.{kind}{self._EXT.get(kind, '.json')}")

    def save(self, job_id: str, kind: str, payload: dict | str | bytes) -> None:
        path = self._path(job_id, kind)
        if isinstance(payload, bytes):
            with open(path, "wb") as f:
                f.write(payload)
        elif isinstance(payload, str):
            with open(path, "w", encoding="utf-8") as f:
                f.write(payload)
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f)

    def load(self, job_id: str, kind: str):
        path = self._path(job_id, kind)
        if kind in self._BINARY:
            with open(path, "rb") as f:
                return f.read()
        with open(path, encoding="utf-8") as f:
            if kind.startswith("bvh"):
                return f.read()
            return json.load(f)

    def path(self, job_id: str, kind: str) -> str:
        """FileResponse용 로컬 경로. S3 전용이면 다운로드 후 경로 반환."""
        p = self._path(job_id, kind)
        if not os.path.exists(p):
            self.load(job_id, kind)  # S3에서 캐시로 가져옴 (없으면 예외)
        return p

    def exists(self, job_id: str, kind: str) -> bool:
        return os.path.exists(self._path(job_id, kind))

    def delete(self, job_id: str) -> None:
        for kind in ("frames", "rig", "rig_mbert", "bvh", "bvh_mbert",
                     "genvideo", "control", "audio"):
            try:
                os.remove(self._path(job_id, kind))
            except OSError:
                pass

    def store_file(self, job_id: str, kind: str, src_path: str) -> str:
        """완성 파일(mp4 등)을 블롭 자리로 이동. 반환: 저장 경로."""
        import shutil
        dst = self._path(job_id, kind)
        shutil.move(src_path, dst)
        return dst


class S3BlobStore(FileBlobStore):
    """쓰기는 S3+로컬, 읽기는 로컬캐시→S3 순."""

    def __init__(self, tmp_dir: str, endpoint: str, bucket: str, prefix: str) -> None:
        super().__init__(tmp_dir)
        self.endpoint = endpoint or None
        self.bucket = bucket
        self.prefix = prefix if prefix.endswith("/") else prefix + "/"
        self._client = None

    def _s3(self):
        if self._client is None:
            import boto3  # lazy: S3 안 쓰면 의존성 불필요
            kw: dict = {}
            if self.endpoint:
                kw["endpoint_url"] = self.endpoint
            import os as _os
            if _os.getenv("AWS_REGION"):
                kw["region_name"] = _os.getenv("AWS_REGION")
            self._client = boto3.client("s3", **kw)
        return self._client

    def _key(self, job_id: str, kind: str) -> str:
        ext = self._EXT.get(kind, ".json")
        return f"{self.prefix}{job_id}.{kind}{ext}"

    def save(self, job_id: str, kind: str, payload: dict | str | bytes) -> None:
        super().save(job_id, kind, payload)
        try:
            if isinstance(payload, bytes):
                body = payload
            elif isinstance(payload, str):
                body = payload.encode("utf-8")
            else:
                body = json.dumps(payload).encode("utf-8")
            ctype = self._MIME.get(kind, "application/json")
            self._s3().put_object(Bucket=self.bucket, Key=self._key(job_id, kind),
                                  Body=body, ContentType=ctype)
        except Exception as e:  # noqa: BLE001
            log.warning("S3 저장 실패, 로컬만 유지: %s", type(e).__name__)

    def load(self, job_id: str, kind: str):
        if super().exists(job_id, kind):
            return super().load(job_id, kind)
        obj = self._s3().get_object(Bucket=self.bucket, Key=self._key(job_id, kind))
        raw = obj["Body"].read()
        super().save(job_id, kind, raw if kind in self._BINARY
                     else raw.decode("utf-8"))
        return super().load(job_id, kind)

    def store_file(self, job_id: str, kind: str, src_path: str) -> str:
        dst = super().store_file(job_id, kind, src_path)
        try:
            ctype = self._MIME.get(kind, "application/json")
            with open(dst, "rb") as f:
                self._s3().put_object(Bucket=self.bucket, Key=self._key(job_id, kind),
                                      Body=f, ContentType=ctype)
        except Exception as e:  # noqa: BLE001
            log.warning("S3 저장 실패, 로컬만 유지: %s", type(e).__name__)
        return dst

    def exists(self, job_id: str, kind: str) -> bool:
        if super().exists(job_id, kind):
            return True
        try:
            self._s3().head_object(Bucket=self.bucket, Key=self._key(job_id, kind))
            return True
        except Exception:  # noqa: BLE001
            return False


def build_blob_store(tmp_dir: str, endpoint: str, bucket: str, prefix: str) -> FileBlobStore:
    if bucket:
        return S3BlobStore(tmp_dir, endpoint, bucket, prefix)
    return FileBlobStore(tmp_dir)
