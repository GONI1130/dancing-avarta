"""Postgres/SQLite 잡 메타데이터. 재시작 후에도 상태가 남는다.

- DATABASE_URL이 postgres://... 이면 psycopg(v3, lazy import) 사용
- sqlite:///path 또는 파일경로/빈값이면 stdlib sqlite3 사용 (추가 의존성 없음)
- 테이블 jobs(id PK, url, status, progress, message, fps, total_frames, created_at, error)

멀티레플리카의 공유 상태 저장소로 Redis 대신 이것만 써도 된다.
(hybrid 모드 = Redis 상태 + Postgres 영속, 둘 다 있으면 Redis 우선·Postgres 미러)
"""
import os
import time
import uuid

from app.store.backends.memory import Job

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs(
 id TEXT PRIMARY KEY, url TEXT, status TEXT, progress INTEGER,
 message TEXT, fps REAL, total_frames INTEGER, created_at REAL, error TEXT)
"""


class PostgresJobs:
    def __init__(self, database_url: str) -> None:
        self.url = database_url.strip()
        self._is_pg = self.url.startswith("postgres")
        if self._is_pg:
            import psycopg  # lazy
            self._pg = psycopg
            self._conn = psycopg.connect(self.url, autocommit=True)
            with self._conn.cursor() as cur:
                cur.execute(_SCHEMA)
        else:
            import sqlite3
            path = self._sqlite_path(self.url)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self._sqlite3 = sqlite3
            # check_same_thread=False: BackgroundTasks 스레드와 공유
            self._conn = sqlite3.connect(path, check_same_thread=False)
            self._conn.execute(_SCHEMA)
            self._conn.commit()
        import threading
        self._lock = threading.Lock()

    @staticmethod
    def _sqlite_path(url: str) -> str:
        if url.startswith("sqlite:///"):
            return url.replace("sqlite:///", "", 1) or "tmp/jobs.db"
        if url.endswith(".db") or url.endswith(".sqlite"):
            return url
        return "tmp/jobs.db"

    def _exec(self, sql: str, params: tuple = ()):
        with self._lock:
            if self._is_pg:
                with self._conn.cursor() as cur:
                    # sqlite ? → postgres %s 변환
                    cur.execute(sql.replace("?", "%s"), params)
                    try:
                        return cur.fetchall()
                    except Exception:  # noqa: BLE001 - DML은 결과 없음
                        return []
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur.fetchall()

    def create(self, url: str) -> Job:
        job = Job(id=str(uuid.uuid4()), url=url)
        self._exec("INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?)",
                   (job.id, job.url, job.status, job.progress, job.message,
                    job.fps, job.total_frames, job.created_at, job.error))
        return job

    def get(self, job_id: str) -> Job | None:
        rows = self._exec("SELECT id,url,status,progress,message,fps,total_frames,created_at,error FROM jobs WHERE id=?", (job_id,))
        if not rows:
            return None
        r = rows[0]
        return Job(id=r[0], url=r[1], status=r[2], progress=int(r[3]),
                   message=r[4] or "", fps=float(r[5]),
                   total_frames=int(r[6]), created_at=float(r[7]), error=r[8] or "")

    def update(self, job_id: str, **kw) -> None:
        allowed = ("url", "status", "progress", "message", "fps", "total_frames", "error")
        sets = [f"{k}=?" for k in kw if k in allowed]
        if not sets:
            return
        vals = [kw[k] for k in kw if k in allowed]
        self._exec(f"UPDATE jobs SET {','.join(sets)} WHERE id=?", tuple(vals) + (job_id,))

    def delete(self, job_id: str) -> None:
        self._exec("DELETE FROM jobs WHERE id=?", (job_id,))

    def list_expired(self, ttl_sec: int) -> list[str]:
        rows = self._exec("SELECT id FROM jobs WHERE created_at<?", (time.time() - ttl_sec,))
        return [r[0] for r in rows]
