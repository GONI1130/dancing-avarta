"""Redis 잡 메타데이터. 멀티레플리카에서 상태 공유용.

키: motion:job:{id} (hash), TTL은 JOB_TTL_SEC.
redis 패키지가 없거나 접속 실패하면 ImportError/ConnectionError를 올려
상위(facade)가 memory로 폴백하게 한다.
"""
import time
import uuid

from app.store.backends.memory import Job


class RedisJobs:
    def __init__(self, url: str, ttl_sec: int) -> None:
        import redis  # lazy
        self._redis = redis.Redis.from_url(url, decode_responses=True)
        self._redis.ping()  # 접속 확인 (실패시 예외 → 폴백)
        self.ttl = ttl_sec
        self._prefix = "motion:job:"

    def _key(self, job_id: str) -> str:
        return self._prefix + job_id

    def create(self, url: str) -> Job:
        job = Job(id=str(uuid.uuid4()), url=url)
        self._redis.hset(self._key(job.id), mapping={
            "url": job.url, "status": job.status, "progress": job.progress,
            "message": job.message, "fps": job.fps,
            "total_frames": job.total_frames,
            "created_at": job.created_at, "error": job.error,
        })
        self._redis.expire(self._key(job.id), self.ttl)
        return job

    def get(self, job_id: str) -> Job | None:
        d = self._redis.hgetall(self._key(job_id))
        if not d:
            return None
        return Job(id=job_id, url=d.get("url", ""), status=d.get("status", "queued"),
                   progress=int(d.get("progress", 0)), message=d.get("message", ""),
                   fps=float(d.get("fps", 0)), total_frames=int(d.get("total_frames", 0)),
                   created_at=float(d.get("created_at", time.time())),
                   error=d.get("error", ""))

    def update(self, job_id: str, **kw) -> None:
        if not kw:
            return
        str_kw = {k: str(v) for k, v in kw.items() if k != "id"}
        self._redis.hset(self._key(job_id), mapping=str_kw)
        self._redis.expire(self._key(job_id), self.ttl)

    def delete(self, job_id: str) -> None:
        self._redis.delete(self._key(job_id))
