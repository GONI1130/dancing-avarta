import os
import uuid
import cv2
import yt_dlp
import mediapipe as mp
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

app = FastAPI(title="Dance Pose Extractor API")

# 프론트엔드 통신을 위한 CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 배포 시 프론트엔드 주소로 변경
    allow_credentials=True,
    allow_methods=["*"],  # GET, POST, OPTIONS 등
    allow_headers=["*"],
)

class YouTubeRequest(BaseModel):
    url: HttpUrl

def cleanup_file(filepath: str):
    """임시 비디오 파일 삭제 함수"""
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            print(f"임시 파일 삭제 완료: {filepath}")
        except Exception as e:
            print(f"파일 삭제 실패: {e}")

def process_video_pose(video_path: str):
    """MediaPipe를 사용해 동영상에서 3D 관절 좌표 추출"""
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,  # 0: Fast, 1: Medium, 2: Heavy (정확도 우선)
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    motion_data = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # BGR -> RGB 변환
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        landmarks_3d = []
        landmarks_2d = []

        if results.pose_world_landmarks:
            for lm in results.pose_world_landmarks.landmark:
                landmarks_3d.append({
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility
                })

        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                landmarks_2d.append({
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility
                })

        # 프레임별 모션 저장
        motion_data.append({
            "frame": frame_idx,
            "timestamp": frame_idx / fps if fps > 0 else 0,
            "poseWorldLandmarks": landmarks_3d,
            "poseLandmarks": landmarks_2d
        })

        frame_idx += 1

    cap.release()
    pose.close()

    return {
        "fps": fps,
        "total_frames": frame_idx,
        "frames": motion_data
    }

@app.post("/api/process-youtube")
async def extract_pose_from_youtube(request: YouTubeRequest, background_tasks: BackgroundTasks):
    url_str = str(request.url)
    task_id = str(uuid.uuid4())
    output_filename = f"temp_{task_id}.mp4"

    # 1. yt-dlp 설정 (720p 이하 mp4로 가볍게 다운로드)
    ydl_opts = {
        'format': 'mp4[height<=720]/bestvideo[height<=720]+bestaudio/best[height<=720]',
        'outtmpl': output_filename,
        'quiet': True,
        'no_warnings': True,
    }

    # 2. 유튜브 영상 다운로드
    try:
        print(f"다운로드 시작: {url_str}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url_str])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"유튜브 다운로드 실패: {str(e)}")

    if not os.path.exists(output_filename):
        raise HTTPException(status_code=500, detail="동영상 파일 생성 실패")

    # 응답 후 백그라운드 작업으로 임시 파일 삭제 예약
    background_tasks.add_task(cleanup_file, output_filename)

    # 3. MediaPipe 관절 추출 실행
    try:
        print("모션 데이터 추출 중...")
        result_data = process_video_pose(output_filename)
        print("모션 추출 완료!")
        return {
            "status": "success",
            "data": result_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모션 추출 실패: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)