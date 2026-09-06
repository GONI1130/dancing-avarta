#!/usr/bin/env bash
set -e

# =========================================================
# 이 스크립트가 하는 일:
#   1) bgutil PO Token 서버(포트 4416)가 안 떠 있으면 백그라운드로 실행
#   2) 정상적으로 뜰 때까지 대기 (ping 체크)
#   3) FastAPI 백엔드(main.py, 포트 8000)를 실행
# =========================================================

# 프로젝트 루트 기준 경로 (필요하면 환경에 맞게 수정하세요)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BGUTIL_DIR="/workspaces/dancing-avarta/bgutil-ytdlp-pot-provider/server"
BGUTIL_LOG="/tmp/bgutil-server.log"
BGUTIL_PORT=4416

echo "=== [1/3] bgutil PO Token 서버 확인 ==="

if curl -s -o /dev/null -w "" "http://127.0.0.1:${BGUTIL_PORT}/ping" 2>/dev/null; then
  echo "이미 실행 중입니다 (포트 ${BGUTIL_PORT}). 건너뜁니다."
else
  if [ ! -d "$BGUTIL_DIR" ]; then
    echo "경고: bgutil 서버 디렉터리를 찾을 수 없습니다: $BGUTIL_DIR"
    echo "설치가 안 되어 있다면 아래 명령으로 먼저 설치하세요:"
    echo "  git clone --single-branch --branch 1.2.2 https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git ~/bgutil-ytdlp-pot-provider"
    echo "  cd ~/bgutil-ytdlp-pot-provider/server && npm install && npx tsc"
    exit 1
  fi

  echo "bgutil 서버 실행 중... (로그: $BGUTIL_LOG)"
  (cd "$BGUTIL_DIR" && nohup node build/main.js > "$BGUTIL_LOG" 2>&1 &)

  echo "=== [2/3] 서버 기동 대기 중 ==="
  MAX_WAIT=20
  waited=0
  until curl -s -o /dev/null -w "" "http://127.0.0.1:${BGUTIL_PORT}/ping" 2>/dev/null; do
    sleep 1
    waited=$((waited + 1))
    if [ "$waited" -ge "$MAX_WAIT" ]; then
      echo "오류: ${MAX_WAIT}초 안에 bgutil 서버가 뜨지 않았습니다."
      echo "로그를 확인하세요: cat $BGUTIL_LOG"
      exit 1
    fi
  done
  echo "bgutil 서버 정상 기동 완료 (${waited}초 소요)"
fi

echo ""
echo "=== [3/3] FastAPI 백엔드 실행 (포트 8000) ==="
cd "$PROJECT_ROOT"
python main.py