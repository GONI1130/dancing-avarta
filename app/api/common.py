"""엔드포인트 공용 헬퍼: 잡 조회, rig/BVH lazy 생성, formats."""
import json

from fastapi import HTTPException

from app.store import store


def require_done_job(job_id: str):
    job = store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status != "done":
        raise HTTPException(status_code=409, detail=f"job not ready: {job.status}")
    return job


def formats_for(job_id: str) -> list[str]:
    fmts = ["full"]
    try:
        if store.bvh_exists(job_id):
            fmts.append("bvh")
    except Exception:  # noqa: BLE001
        pass
    try:
        if store.video_exists(job_id, "control"):
            fmts.append("control")
        if store.video_exists(job_id, "genvideo"):
            fmts.append("video")
        if store.video_exists(job_id, "audio"):
            fmts.append("audio")
    except Exception:  # noqa: BLE001
        pass
    return fmts


def _from_json_maybe(payload):
    if isinstance(payload, str):
        return json.loads(payload)
    return payload


def ensure_rig(job_id: str) -> dict:
    """rig 캐시 반환. 없으면 frames에서 생성 (구 job 호환). BVH 내부용."""
    if not store.rig_exists(job_id, kind="rig"):
        from app.services.exports import to_rig
        rig = to_rig(_from_json_maybe(store.blob.load(job_id, "frames")))
        store.save_rig(job_id, rig)
    if not store.rig_exists(job_id, kind="rig"):
        raise FileNotFoundError("frames expired, re-request")
    return _from_json_maybe(store.blob.load(job_id, "rig"))


def ensure_bvh_text(job_id: str) -> str:
    """BVH 텍스트 반환 (없으면 rig에서 lazy 생성)."""
    if not store.bvh_exists(job_id):
        from app.services.exports import to_bvh
        store.save_bvh(job_id, to_bvh(ensure_rig(job_id)))
    return store.load_bvh(job_id)


def load_frames_blob(job_id: str) -> dict:
    return _from_json_maybe(store.blob.load(job_id, "frames"))
