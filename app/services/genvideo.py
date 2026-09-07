"""참고용 AI 춤 영상 백엔드. 캐릭터 사진 1장 + 모션 영상 → 춤추는 영상 mp4.

아바타 옆 참고 패널용이며 결과물이 아님. 아바타가 주 출력, 이 영상은
춤이 어떻게 보여야 하는지 대조용으로만 쓴다.

GENVIDEO_BACKEND=mock(기본, 자리표시자) | replicate(MimicMotion API) | local(자체 GPU, 미구현 안내)
- mock: 캐릭터 사진 위에 스켈레톤을 합성한 플레이스홀더. 파이프·API·프론트
  검증용이며 실제 확산 생성은 아님.
- replicate: REPLICATE_API_TOKEN 필요. 모델 GENVIDEO_MODEL
  (기본 zsxkib/mimic-motion). 입력 키는 환경으로 재정의 가능
  (플레이그라운드 스키마 확인 권장).
- local: 공식 MimicMotion Docker를 직접 띄우는 경우. 안내 에러만 던짐.
"""
import os
import shutil
import urllib.request


def generate(character_image: str, motion_video: str, out_path: str,
             progress_cb=None) -> str:
    backend = os.getenv("GENVIDEO_BACKEND", "mock").lower()
    if backend == "replicate":
        return _via_replicate(character_image, motion_video, out_path, progress_cb)
    if backend == "local":
        raise RuntimeError(
            "GENVIDEO_BACKEND=local은 자체 GPU 서버가 필요합니다: "
            "공식 MimicMotion Docker(https://github.com/tencent/MimicMotion)를 띄우고 "
            "GENVIDEO_BACKEND=replicate 또는 mock을 사용하세요")
    return _via_mock(character_image, motion_video, out_path, progress_cb)


def fallback_character(video_path: str, payload: dict, out_path: str) -> str:
    """캐릭터 사진이 없을 때: 첫 유효 프레임의 인물 영역을 정사각 크롭."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("캐릭터 폴백용 영상 열기 실패")
    try:
        target = None
        for f in payload.get("frames", []):
            if f.get("poseLandmarks"):
                target = f
                break
        if target is None:
            raise RuntimeError("캐릭터 사진이 필요합니다 (인물 검출 0명)")
        xs = [float(lm.get("x", 0.5)) for lm in target["poseLandmarks"]]
        ys = [float(lm.get("y", 0.5)) for lm in target["poseLandmarks"]]
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("첫 프레임 읽기 실패")
        h, w = frame.shape[:2]
        x0, x1 = max(min(xs) - 0.1, 0) * w, min(max(xs) + 0.1, 1) * w
        y0, y1 = max(min(ys) - 0.1, 0) * h, min(max(ys) + 0.1, 1) * h
        side = max(x1 - x0, y1 - y0, 10)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        x0, y0 = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0))
        x1, y1 = int(min(cx + side / 2, w)), int(min(cy + side / 2, h))
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        cv2.imwrite(out_path, frame[y0:y1, x0:x1])
        return out_path
    finally:
        cap.release()


def _via_mock(character_image: str, motion_video: str, out_path: str,
              progress_cb=None) -> str:
    """자리표시자: 캐릭터 사진 배경 + 모션 영상 합성. 형식만 mp4."""
    import cv2
    char = cv2.imread(character_image)
    if char is None:
        raise RuntimeError(f"캐릭터 사진 읽기 실패: {character_image}")
    cap = cv2.VideoCapture(motion_video)
    if not cap.isOpened():
        raise RuntimeError(f"모션 영상 열기 실패: {motion_video}")
    fps = cap.get(5) or 30.0
    w = int(cap.get(3)) or 512
    h = int(cap.get(4)) or 512
    char = cv2.resize(char, (w, h))
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError("생성 영상 writer 생성 실패")
    n = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # 어두운 모션 실루엣을 캐릭터 사진 위에 반투명 합성
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
            mask3 = cv2.merge([mask, mask, mask]) // 255
            bold = cv2.dilate(frame, None, iterations=2)
            comp = (char * 0.55 + bold * 0.45).astype("uint8")
            comp = (comp * (0.35 + 0.65 * (mask3 > 0))).astype("uint8")
            cv2.putText(comp, "MOCK PREVIEW - set GENVIDEO_BACKEND=replicate",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            vw.write(comp)
            n += 1
            if progress_cb and n % 30 == 0:
                progress_cb(n, 0)
    finally:
        cap.release()
        vw.release()
    if n == 0:
        raise RuntimeError("모션 영상 프레임 없음")
    return out_path


def _via_replicate(character_image: str, motion_video: str, out_path: str,
                   progress_cb=None) -> str:
    try:
        import replicate
    except ImportError as e:
        raise RuntimeError("replicate 패키지 필요: pip install replicate "
                           "+ REPLICATE_API_TOKEN 설정") from e
    token = os.getenv("REPLICATE_API_TOKEN", "").strip()
    if not token:
        raise RuntimeError("REPLICATE_API_TOKEN이 비어 있습니다")
    model = os.getenv("GENVIDEO_MODEL", "zsxkib/mimic-motion")
    img_key = os.getenv("GENVIDEO_INPUT_IMAGE_KEY", "ref_image")
    vid_key = os.getenv("GENVIDEO_INPUT_VIDEO_KEY", "ref_video")
    if progress_cb:
        progress_cb(0, 1)
    with open(character_image, "rb") as fi, open(motion_video, "rb") as fv:
        output = replicate.run(model, input={img_key: fi, vid_key: fv})
    url = output if isinstance(output, str) else (output[-1] if output else None)
    if not url:
        raise RuntimeError("Replicate 응답에 영상 URL 없음 (입력 키 확인 필요)")
    tmp_dl = out_path + ".downloading"
    urllib.request.urlretrieve(url
                               if isinstance(url, str) else str(url), tmp_dl)
    shutil.move(tmp_dl, out_path)
    if progress_cb:
        progress_cb(1, 1)
    return out_path
