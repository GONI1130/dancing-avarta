"""추출된 2D 관절 → 스켈레톤 컨트롤 영상(mp4) 렌더러.

생성형 영상(MimicMotion 등)의 모션 가이드·미리보기용. mediapipe drawing에
의존하지 않고 33관절 연결관계만 내장 → 추가 의존성 없음.
"""
import os

# MediaPipe Pose 33 연결관계 (양손 17-22 제외한 몸통 위주 + 손목 연결)
POSE_LINKS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (27, 31), (28, 30), (28, 32),
]


def render_control_video(payload: dict, out_path: str, height: int = 768,
                         progress_cb=None) -> str:
    """frames[].poseLandmarks → 검은 배경 스켈레톤 mp4. 반환: 경로."""
    import cv2
    import numpy as np

    frames = payload.get("frames", [])
    fps = payload.get("fps", 30.0) or 30.0
    iw, ih = float(payload.get("img_w", 16) or 16), float(payload.get("img_h", 9) or 9)
    w = max(64, int(height * iw / max(ih, 1e-6)))
    h = height
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError("컨트롤 영상 writer 생성 실패")
    try:
        for idx, f in enumerate(frames):
            img = np.zeros((h, w, 3), np.uint8)
            lms = f.get("poseLandmarks") or []
            pts = []
            for lm in lms:
                try:
                    x = float(lm.get("x", -1)) * w
                    y = float(lm.get("y", -1)) * h
                    v = float(lm.get("visibility", 0))
                except (TypeError, ValueError, AttributeError):
                    x, y, v = -1, -1, 0.0
                pts.append((x, y, v))
            for a, b in POSE_LINKS:
                if a < len(pts) and b < len(pts) and pts[a][2] > 0.3 and pts[b][2] > 0.3:
                    # 신뢰도를 밝기로 (MimicMotion식 confidence-aware와同思想)
                    bright = int(120 + 135 * min(pts[a][2], pts[b][2]))
                    cv2.line(img, (int(pts[a][0]), int(pts[a][1])),
                             (int(pts[b][0]), int(pts[b][1])), (bright, bright, bright), 3)
            for (x, y, v) in pts:
                if v > 0.3 and 0 <= x < w and 0 <= y < h:
                    cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)
            vw.write(img)
            if progress_cb and idx % 30 == 0:
                progress_cb(idx, len(frames))
    finally:
        vw.release()
    return out_path


def save_b64_image(b64: str, out_path: str) -> str:
    """base64(dataURL 허용) → jpg 파일. 캐릭터 사진 저장용."""
    import base64
    import cv2
    import numpy as np
    s = (b64 or "").strip()
    if "," in s and s.startswith("data:"):
        s = s.split(",", 1)[1]
    try:
        raw = base64.b64decode(s)
    except Exception as e:
        raise RuntimeError(f"캐릭터 사진 base64 해석 실패: {e}") from e
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("캐릭터 사진 해석 실패")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, img)
    return out_path
