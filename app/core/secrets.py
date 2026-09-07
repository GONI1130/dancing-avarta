"""쿠키/API키 시크릿 해결기.

우선순위:
1. 환경변수 COOKIES_B64 (base64 인코딩된 cookies.txt 전체)
2. Docker secret 파일 COOKIES_SECRET_FILE (/run/secrets/youtube_cookies)
3. 기존 COOKIES_PATH 파일 (로컬 개발용)

yt-dlp는 파일 경로를 요구하므로 1/2번은 tmp/cookies.materialized.txt로
실체화(materialize)해서 경로를 반환한다. 원본 시크릿은 절대 로그에 찍지 않는다.
"""
import base64
import logging
import os

log = logging.getLogger(__name__)


def load_api_keys(api_keys: list[str], api_key_file: str) -> set[str]:
    keys: set[str] = set(k.strip() for k in api_keys if k.strip())
    if api_key_file and os.path.exists(api_key_file):
        try:
            with open(api_key_file, encoding="utf-8") as f:
                for line in f.read().split(","):
                    line = line.strip()
                    if line:
                        keys.add(line)
        except OSError as e:
            log.warning("API_KEY_FILE 읽기 실패: %s", e)
    return keys


def ensure_cookies_file(cookies_b64: str, secret_file: str, cookies_path: str, tmp_dir: str) -> str:
    """yt-dlp에 넘길 쿠키 파일 경로를 보장. 없으면 빈 문자열 반환(쿠키 없이 시도)."""
    os.makedirs(tmp_dir, exist_ok=True)
    materialized = os.path.join(tmp_dir, "cookies.materialized.txt")

    if cookies_b64.strip():
        try:
            raw = base64.b64decode(cookies_b64)
            with open(materialized, "wb") as f:
                f.write(raw)
            return materialized
        except Exception as e:  # noqa: BLE001
            log.warning("COOKIES_B64 디코드 실패, 다음 소스로 폴백: %s", type(e).__name__)

    if secret_file and os.path.exists(secret_file):
        try:
            with open(secret_file, "rb") as src, open(materialized, "wb") as dst:
                dst.write(src.read())
            return materialized
        except OSError as e:
            log.warning("secret 파일 복사 실패: %s", e)

    if cookies_path and os.path.exists(cookies_path):
        return cookies_path
    return ""
