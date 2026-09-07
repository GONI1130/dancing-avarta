"""진행률 푸시 WebSocket."""
import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.store import store

router = APIRouter(prefix="/api/v1/motions", tags=["motions"])


@router.websocket("/{job_id}/ws")
async def motion_job_ws(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    try:
        while True:
            job = store.get(job_id)
            if not job:
                await websocket.send_json({"error": "job not found"})
                break
            await websocket.send_json({
                "job_id": job.id, "status": job.status, "progress": job.progress,
                "message": job.message, "fps": job.fps,
                "total_frames": job.total_frames, "error": job.error,
            })
            if job.status in ("done", "error"):
                break
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
