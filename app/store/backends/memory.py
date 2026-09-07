"""인메모리 잡 메타데이터. 로컬 개발/단일레플리카 기본값."""
import threading
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class Job:
    id: str
    url: str
    status: str = "queued"
    progress: int = 0
    message: str = ""
    fps: float = 0.0
    total_frames: int = 0
    created_at: float = field(default_factory=time.time)
    error: str = ""


class MemoryJobs:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}

    def create(self, url: str) -> Job:
        job = Job(id=str(uuid.uuid4()), url=url)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **kw) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                for k, v in kw.items():
                    if hasattr(job, k):
                        setattr(job, k, v)

    def list_expired(self, ttl_sec: int) -> list[str]:
        now = time.time()
        with self._lock:
            return [jid for jid, j in self._jobs.items() if now - j.created_at > ttl_sec]

    def delete(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)
