"""기준 얼굴사진 ↔ 영상 속 얼굴 매칭 (대상 지정용).

InsightFace(ArcFace, buffalo_l, CPU)를 lazy import. 없으면 설치 안내 에러.
같은 사람 판정은 코사인 유사도 ≥ FACE_THRESHOLD(기본 0.4).
"""
import base64

import cv2
import numpy as np

_app = None


def _get_app():
    global _app
    if _app is not None:
        return _app
    try:
        from insightface.app import FaceAnalysis
    except ImportError as e:
        raise RuntimeError(
            "target=face에는 insightface가 필요합니다: pip install insightface "
            "(CPU용 onnxruntime 포함, 모델 buffalo_l 자동 다운로드)") from e
    _app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    _app.prepare(ctx_id=-1, det_size=(320, 320))
    return _app


def decode_ref_image(ref_b64: str):
    """base64(dataURL 허용) → RGB ndarray. 실패시 RuntimeError."""
    try:
        s = ref_b64.strip()
        if "," in s and s.startswith("data:"):
            s = s.split(",", 1)[1]
        raw = base64.b64decode(s)
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("decode null")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        raise RuntimeError(f"기준 얼굴사진 해석 실패: {e}") from e


def embed_face_bgr(rgb_crop) -> np.ndarray | None:
    """얼굴 크롭(RGB) → 정규화 임베딩. 얼굴 없으면 None."""
    app = _get_app()
    faces = app.get(rgb_crop)
    if not faces:
        return None
    emb = np.array(sorted(faces, key=lambda f: -f.det_score)[0].normed_embedding,
                   dtype=np.float32)
    n = float(np.linalg.norm(emb))
    return emb / n if n > 1e-9 else None


def face_crop_rgb(frame_rgb, lm2d, margin: float = 0.6):
    """포즈 2D(코0·귀7,8)에서 얼굴 박스 크롭. 불가시 None."""
    try:
        h, w = frame_rgb.shape[:2]
        pts = []
        for i in (0, 7, 8):
            p = lm2d[i]
            x, y = (float(p.get("x", -1)), float(p.get("y", -1))) if isinstance(p, dict) \
                else (float(p[0]), float(p[1]))
            if 0 <= x <= 1 and 0 <= y <= 1:
                pts.append((x * w, y * h))
        if len(pts) < 2:
            return None
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        side = max(max(xs) - min(xs), max(ys) - min(ys), 20.0) * (1 + margin)
        x0, y0 = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0))
        x1, y1 = int(min(cx + side / 2, w)), int(min(cy + side / 2, h))
        if x1 - x0 < 30 or y1 - y0 < 30:
            return None
        return frame_rgb[y0:y1, x0:x1]
    except (IndexError, TypeError, ValueError):
        return None


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def match_candidates(crops: list, ref_emb: np.ndarray) -> tuple[int | None, float]:
    """후보 크롭들 중 기준 얼굴과 가장 유사한 인덱스·유사도. 전부 실패시 (None, 0.0)."""
    best, best_sim = None, 0.0
    for i, crop in enumerate(crops):
        if crop is None:
            continue
        try:
            emb = embed_face_bgr(crop)
        except Exception:  # noqa: BLE001 - 한 후보 실패가 전체를 막으면 안 됨
            continue
        if emb is None:
            continue
        sim = cosine(emb, ref_emb)
        if sim > best_sim:
            best, best_sim = i, sim
    return best, round(best_sim, 3)
