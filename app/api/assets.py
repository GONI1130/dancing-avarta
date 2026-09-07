"""파일 자산: /bvh, /video, /control, /audio."""
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse

from app.api.common import ensure_bvh_text, require_done_job
from app.api.security import check_rate_limit, require_api_key
from app.store import store

router = APIRouter(prefix="/api/v1/motions", tags=["motions"])


@router.get("/{job_id}/bvh", response_class=PlainTextResponse,
            dependencies=[Depends(require_api_key)])
def get_motion_bvh(job_id: str, request: Request) -> PlainTextResponse:
    check_rate_limit(request, "read")
    require_done_job(job_id)
    try:
        text = ensure_bvh_text(job_id)
        return PlainTextResponse(text, media_type="text/plain",
                                 headers={"Content-Disposition": f"attachment; filename=motion_{job_id}.bvh"})
    except FileNotFoundError:
        raise HTTPException(status_code=410, detail="frames expired, re-request")


def _mp4_or(job_id: str, kind: str, missing_detail: str, filename: str) -> FileResponse:
    if not store.video_exists(job_id, kind):
        raise HTTPException(status_code=409, detail=missing_detail)
    return FileResponse(store.video_path(job_id, kind), media_type="video/mp4",
                        filename=filename)


@router.get("/{job_id}/video", dependencies=[Depends(require_api_key)])
def get_motion_video(job_id: str, request: Request):
    """참고용 AI 춤 영상 mp4 (아바타 옆 대조용)."""
    check_rate_limit(request, "read")
    require_done_job(job_id)
    return _mp4_or(job_id, "genvideo",
                   "video not ready (render=ref로 생성 필요)",
                   f"dance_{job_id}.mp4")


@router.get("/{job_id}/control", dependencies=[Depends(require_api_key)])
def get_control_video(job_id: str, request: Request):
    """스켈레톤 컨트롤 영상 미리보기."""
    check_rate_limit(request, "read")
    require_done_job(job_id)
    return _mp4_or(job_id, "control", "control not ready", f"control_{job_id}.mp4")


@router.get("/{job_id}/audio", dependencies=[Depends(require_api_key)])
def get_motion_audio(job_id: str, request: Request):
    """원본 음원(m4a). 아바타 재생과 같은 길이라 싱크가 맞음."""
    check_rate_limit(request, "read")
    require_done_job(job_id)
    if not store.video_exists(job_id, "audio"):
        raise HTTPException(status_code=409, detail="audio not ready (ffmpeg 없음 등)")
    return FileResponse(store.video_path(job_id, "audio"), media_type="audio/mp4",
                        filename=f"audio_{job_id}.m4a")
