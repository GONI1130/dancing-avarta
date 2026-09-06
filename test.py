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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
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

def process_video_pose(video_path: str, output_viz_path: str = None):
    """MediaPipe를 사용해 관절을 추출하고, 필요시 스켈레톤이 그려진 동영상을 생성"""
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils          # 관절 그리기 도구
    mp_drawing_styles = mp.solutions.drawing_styles  # 스타일 설정

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 비디오 저장 설정 (시각화 동영상 생성용)
    out_video = None
    if output_viz_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(output_viz_path, fourcc, fps, (width, height))

    motion_data = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # =========================================================
        # 원본 영상 프레임 위에 MediaPipe 뼈대(Skeleton) 그리기
        # =========================================================
        if results.pose_landmarks and out_video:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            out_video.write(frame)  # 뼈대가 그려진 프레임을 동영상에 저장

        # 3D/2D 관절 좌표 추출 로직 (기존 동일)
        landmarks_3d = []
        landmarks_2d = []

        if results.pose_world_landmarks:
            for lm in results.pose_world_landmarks.landmark:
                landmarks_3d.append({"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility})

        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                landmarks_2d.append({"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility})

        motion_data.append({
            "frame": frame_idx,
            "timestamp": frame_idx / fps if fps > 0 else 0,
            "poseWorldLandmarks": landmarks_3d,
            "poseLandmarks": landmarks_2d
        })

        frame_idx += 1

    cap.release()
    if out_video:
        out_video.release()
    pose.close()

    return {
        "fps": fps,
        "total_frames": frame_idx,
        "frames": motion_data
    }
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    motion_data = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

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
    viz_filename = f"result_{task_id}.mp4"  # 관절 영상 저장 경로

    # =========================================================
    # 유튜브 봇 차단 회피 옵션이 추가된 yt-dlp 설정
    # =========================================================
    base_opts = {
        # AV1(av01)은 서버에 하드웨어 디코더가 없어 소프트웨어 디코딩 중
        # 프레임 손실이 발생할 수 있으므로, H.264(avc1) 코덱을 우선 선택한다.
        # 최우선: avc1 비디오 + m4a 오디오 → 없으면 avc1 비디오만이라도 → 그래도 없으면 기존 방식
        'format': 'bestvideo[vcodec^=avc1]+bestaudio[acodec^=mp4a]/bestvideo[vcodec^=avc1]+bestaudio/bestvideo*+bestaudio/best',
        'merge_output_format': 'mp4',
        'outtmpl': output_filename,
        'quiet': False,
        'no_warnings': False,
        'verbose': True,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        # set 타입으로 지정 (문자열로 주면 한 글자씩 쪼개져 인식됨)
        'remote_components': {'ejs:github'},
    }

    # 1차 시도: 쿠키 없이 android/ios 클라이언트 (공개 영상은 대부분 이걸로 충분)
    attempt_1 = {
        **base_opts,
        'extractor_args': {'youtube': {'player_client': ['android', 'ios']}},
    }

    # 2차 시도(폴백): 쿠키 + 여러 클라이언트 동시 지정
    # (web 단독은 유튜브의 SABR 강제 실험에 걸리면 실패하므로, tv/mweb도 함께 시도해
    #  그중 살아있는 포맷을 내려주는 클라이언트를 쓰도록 폭을 넓힌다)
    attempt_2 = {
        **base_opts,
        'cookiefile': 'cookies.txt',
        'extractor_args': {'youtube': {'player_client': ['web', 'tv', 'mweb']}},
    }

    last_error = None
    for attempt_name, opts in (("android/ios (쿠키 없음)", attempt_1), ("web (쿠키 사용)", attempt_2)):
        try:
            print(f"다운로드 시작 [{attempt_name}]: {url_str}", flush=True)
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url_str])
            last_error = None
            break
        except Exception as e:
            print(f"[{attempt_name}] 실패: {e}", flush=True)
            last_error = e
            continue

    if last_error is not None:
        raise HTTPException(status_code=400, detail=f"유튜브 다운로드 실패: {str(last_error)}")

    if not os.path.exists(output_filename):
        raise HTTPException(status_code=500, detail="동영상 파일 생성 실패")

    background_tasks.add_task(cleanup_file, output_filename)

    try:
        print("모션 데이터 추출 및 시각화 영상 생성 중...")
        # 시각화 영상 파일명을 인자로 전달
        result_data = process_video_pose(output_filename, output_viz_path=viz_filename)
        print(f"관절 시각화 영상 생성 완료: {viz_filename}")

        return {
            "status": "success",
            "data": result_data,
            "viz_video": viz_filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모션 추출 실패: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)