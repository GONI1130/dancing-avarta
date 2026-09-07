"""구 클라이언트(index.html 구버전) 호환용 동기 shim.

신규 앱은 /api/v1/motions를 쓸 것. 이 엔드포인트는 점진적으로 제거 예정.
"""
import os
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.core.config import get_settings
from app.core.secrets import ensure_cookies_file
from app.schemas.motion import YouTubeRequest
from app.services.pose import process_video_pose
from app.services.youtube import download_youtube

router = APIRouter(tags=["legacy"])


@router.post("/api/process-youtube")
def legacy_process_youtube(req: YouTubeRequest, background_tasks: BackgroundTasks) -> dict:
    settings = get_settings()
    cookies = ensure_cookies_file(settings.cookies_b64, settings.cookies_secret_file,
                                  settings.cookies_path, settings.tmp_dir)
    task_id = str(uuid.uuid4())
    out_path = os.path.join(settings.tmp_dir, f"temp_{task_id}.mp4")
    try:
        video_path = download_youtube(str(req.url), out_path, cookies_path=cookies)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"유튜브 다운로드 실패: {e}") from e
    background_tasks.add_task(lambda p=out_path: os.path.exists(p) and os.remove(p))
    try:
        result = process_video_pose(video_path, max_frames=settings.max_frames)
        try:
            from app.services.kinematics import (
                attach_fk_joints,
                attach_foot_joints,
                attach_hand_joints,
                attach_neck_head,
                attach_root,
            )
            attach_hand_joints(result)
            attach_foot_joints(result)
            attach_neck_head(result)
            attach_fk_joints(result)
            attach_root(result)
        except Exception:  # noqa: BLE001
            pass
        return {"status": "success", "data": result}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"모션 추출 실패: {e}") from e
