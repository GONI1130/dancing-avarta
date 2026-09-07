"""인증 + 레이트리밋. (core/auth.py + core/ratelimit.py 통합)

- API_KEYS(또는 API_KEY_FILE 시크릿)가 비어있으면 개발모드: 인증 스킵 + 경고 1회.
- 있으면 X-API-Key 또는 ?api_key 필요.
- 생성/조회는 분당 한도. 초과시 429 + Retry-After.
"""
import logging
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request

from app.core.config import get_settings
from app.core.secrets import load_api_keys

log = logging.getLogger(__name__)
_warned_open = False


class RateLimiter:
    """의존성 없는 인메모리 토큰버킷. 멀티레플리카에선 각자 제한(안전 방향)."""

    def __init__(self) -> None:
        # key -> deque[timestamps]
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str, limit_per_min: int) -> tuple[bool, int]:
        """(허용여부, 남은횟수). limit<=0이면 무제한."""
        if limit_per_min <= 0:
            return True, -1
        now = time.time()
        window = self._hits[key]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= limit_per_min:
            return False, 0
        window.append(now)
        return True, limit_per_min - len(window)


limiter = RateLimiter()


def _active_keys() -> set[str]:
    s = get_settings()
    return load_api_keys(s.api_keys, s.api_key_file)


async def require_api_key(request: Request,
                          x_api_key: str | None = Header(default=None, alias="X-API-Key"),
                          api_key_q: str | None = None) -> None:
    global _warned_open
    keys = _active_keys()
    if not keys:
        if not _warned_open:
            log.warning("API_KEYS 미설정: 개발모드로 인증 우회 중. 운영에선 반드시 설정.")
            _warned_open = True
        return
    if ((x_api_key or api_key_q or "").strip()) in keys:
        return
    raise HTTPException(status_code=401, detail="invalid api key")


def check_rate_limit(request: Request, kind: str) -> None:
    s = get_settings()
    limit = s.rate_limit_create_per_min if kind == "create" else s.rate_limit_read_per_min
    if limit <= 0:
        return
    ident = request.client.host if request.client else "unknown"
    ok, _ = limiter.check(f"{kind}:{ident}", limit)
    if not ok:
        raise HTTPException(status_code=429, detail="rate limit exceeded",
                            headers={"Retry-After": "60"})
