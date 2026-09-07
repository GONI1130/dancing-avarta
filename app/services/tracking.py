"""대상 지정·프레임 추적. 검출(detect.py)과 분리.

Tracker: 첫 유효 프레임에서 선택 → 이후 최근접 중심+크기로 추적.
게이트 이탈시 공백 허용(TRACK_GAP_TOL) 후 재선택.
"""
import os


def _d2(lm, i):
    """2D dict 리스트에서 (x, y, visibility)."""
    try:
        p = lm[i]
        if isinstance(p, dict):
            return (float(p.get("x", 0.5)), float(p.get("y", 0.5)), float(p.get("visibility", 0)))
        return (float(p[0]), float(p[1]), float(p[3]) if len(p) > 3 else 1.0)
    except (IndexError, TypeError, ValueError):
        return (0.5, 0.5, 0.0)


def _person_entries(lms2d_list, lms3d_list) -> list[dict]:
    """프레임의 N명 → [{center, size, score, lms2d, lms3d}]."""
    out = []
    for k, lms2d in enumerate(lms2d_list or []):
        if not lms2d or len(lms2d) < 29:
            continue
        lms3d = lms3d_list[k] if lms3d_list and k < len(lms3d_list) else []
        hx = (_d2(lms2d, 23)[0] + _d2(lms2d, 24)[0]) / 2
        hy = (_d2(lms2d, 23)[1] + _d2(lms2d, 24)[1]) / 2
        sx = (_d2(lms2d, 11)[0] + _d2(lms2d, 12)[0]) / 2
        sy = (_d2(lms2d, 11)[1] + _d2(lms2d, 12)[1]) / 2
        size = max(((sx - hx) ** 2 + (sy - hy) ** 2) ** 0.5, 1e-6)
        score = sum(_d2(lms2d, i)[2] for i in (11, 12, 23, 24)) / 4
        out.append({"center": (hx, hy), "size": size, "score": score,
                    "lms2d": lms2d, "lms3d": lms3d})
    return out


def _pick_initial(persons: list, mode: str) -> int | None:
    """위치 힌트로 첫 대상 선택. left=화면왼쪽, right=오른쪽, 그 외=중앙."""
    if not persons:
        return None
    if mode == "left":
        return min(range(len(persons)), key=lambda i: persons[i]["center"][0])
    if mode == "right":
        return max(range(len(persons)), key=lambda i: persons[i]["center"][0])
    return min(range(len(persons)), key=lambda i: abs(persons[i]["center"][0] - 0.5))


def _track_cost(p: dict, prev: dict) -> float:
    dc = ((p["center"][0] - prev["center"][0]) ** 2
          + (p["center"][1] - prev["center"][1]) ** 2) ** 0.5
    ds = abs(p["size"] - prev["size"]) / max(prev["size"], 1e-6)
    return dc + 0.5 * ds


def _wrist_mid_2d(lms2d):
    try:
        return ((_d2(lms2d, 15)[0] + _d2(lms2d, 16)[0]) / 2,
                (_d2(lms2d, 15)[1] + _d2(lms2d, 16)[1]) / 2)
    except (IndexError, TypeError):
        return (0.5, 0.5)


def assign_hands(hand_entries: list, persons: list, target_pos: int | None) -> list:
    """손들을 가장 가까운 사람의 손목에 할당, 대상 사람 것만 반환."""
    if target_pos is None or not persons or target_pos >= len(persons):
        return hand_entries if len(persons) <= 1 else []
    if len(persons) <= 1:
        return hand_entries
    wrists = [_wrist_mid_2d(p["lms2d"]) for p in persons]
    out = []
    for h in hand_entries:
        try:
            lms = h.get("landmarks") or []
            wx = float(lms[0].get("x", 0.5))
            wy = float(lms[0].get("y", 0.5))
        except (IndexError, TypeError, ValueError, AttributeError):
            continue
        best = min(range(len(wrists)),
                   key=lambda i: (wrists[i][0] - wx) ** 2 + (wrists[i][1] - wy) ** 2)
        d = ((wrists[best][0] - wx) ** 2 + (wrists[best][1] - wy) ** 2) ** 0.5
        if best == target_pos and d < 0.25:
            out.append(h)
    return out


def match_face(persons: list, frame_rgb, ref_emb, threshold: float) -> tuple[int | None, float]:
    """첫 유효 프레임에서 후보 얼굴 크롭 매칭. (인덱스, 유사도)."""
    from app.services.face_match import face_crop_rgb, match_candidates
    crops = []
    for p in persons:
        try:
            crops.append(face_crop_rgb(frame_rgb, p["lms2d"]))
        except Exception:  # noqa: BLE001
            crops.append(None)
    best, sim = match_candidates(crops, ref_emb)
    if best is None or sim < threshold:
        return None, sim
    return best, sim


class Tracker:
    """한 영상의 대상 추적 상태. update()마다 (lms3d, lms2d, pick, lost) 반환."""

    def __init__(self, mode: str = "auto") -> None:
        self.mode = mode
        self.gate = float(os.getenv("TRACK_GATE", "0.25"))
        self.gap_tol = int(os.getenv("TRACK_GAP_TOL", "15"))
        self.face_thr = float(os.getenv("FACE_THRESHOLD", "0.4"))
        self.prev = None
        self.gap = 0
        self.persons_max = 0
        self.label = mode if mode != "face" else "face:?"
        self.face_matched = False

    def update(self, persons: list, frame_rgb=None, ref_emb=None):
        self.persons_max = max(self.persons_max, len(persons))
        pick, lost = None, False
        if persons:
            if self.prev is None or self.gap > self.gap_tol:
                if self.mode == "face" and not self.face_matched:
                    pick, sim = match_face(persons, frame_rgb, ref_emb, self.face_thr)
                    if pick is not None:
                        self.label = f"face:{sim:.2f}"
                        self.face_matched = True
                    else:
                        pick = _pick_initial(persons, "center")
                        self.label = "face:miss→center"
                else:
                    pick = _pick_initial(persons, self.mode)
                self.gap = 0
            else:
                costs = [_track_cost(p, self.prev) for p in persons]
                best = min(range(len(costs)), key=lambda i: costs[i])
                if costs[best] <= self.gate:
                    pick = best
                    self.gap = 0
                else:
                    lost = True
                    self.gap += 1
        else:
            lost = True
            self.gap += 1
        if pick is not None:
            self.prev = persons[pick]
            return persons[pick]["lms3d"], persons[pick]["lms2d"], pick, lost
        return [], [], None, lost
