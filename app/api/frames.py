"""프레임 조회 (full 단일)."""
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.api.common import load_frames_blob, require_done_job
from app.api.security import check_rate_limit, require_api_key
from app.core.config import get_settings
from app.store import store

router = APIRouter(prefix="/api/v1/motions", tags=["motions"])


@router.get("/{job_id}/frames", dependencies=[Depends(require_api_key)])
def get_motion_frames(job_id: str, request: Request,
                      page: int = Query(0, ge=0),
                      page_size: int = Query(60, ge=1, le=120),
                      stride: int = Query(1, ge=1, le=10)) -> dict:
    check_rate_limit(request, "read")
    require_done_job(job_id)
    try:
        return store.paginate(load_frames_blob(job_id), page,
                              min(page_size, get_settings().frames_page_max), stride)
    except FileNotFoundError:
        raise HTTPException(status_code=410, detail="frames expired, re-request")
