"""관절 수학: 방향벡터→오일러, 몸통 11관절·손가락·루트모션 계산.

입출력 포맷(rig/compact/BVH)은 exports.py. 둘로 나뉘기 전 retarget.py였다.
"""
import math
import os

# 좌표계 전제 (실측): MediaPipe world/2D 모두 y-아래+ (코 -0.6, 발목 +0.7).
# 따라서 화면-아래 방향 = (0,+1,0), 화면-위 = (0,-1,0). ref는 이 기준이다.
DOWN = (0.0, 1.0, 0.0)
UP = (0.0, -1.0, 0.0)

# MediaPipe Pose 인덱스
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 23, 24, 25, 26, 27, 28
L_HEEL, R_HEEL, L_FOOT, R_FOOT = 29, 30, 31, 32
NOSE = 0

RIG_JOINTS = ("Spine", "Chest", "Neck", "Head", "Hips",
              "LeftUpperArm", "LeftLowerArm", "RightUpperArm", "RightLowerArm",
              "LeftUpperLeg", "LeftLowerLeg", "RightUpperLeg", "RightLowerLeg",
              "LeftFoot", "RightFoot", "LeftToe", "RightToe")
# 발 정면 rest: 카메라 쪽(+z) 수평. 서서 정면 기준 근사치.
FWD = (0.0, 0.0, 1.0)


def _v(a, b):
    return (b[0] - a[0], b[1] - a[1], b[2] - a[2])


def _norm(v):
    n = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    return (v[0] / n, v[1] / n, v[2] / n) if n > 1e-9 else (0.0, 1.0, 0.0)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _quat_from_to(a, b):
    """단위벡터 a→b 최단호 quaternion (x,y,z,w)."""
    d = max(-1.0, min(1.0, _dot(a, b)))
    if d > 0.999999:
        return (0.0, 0.0, 0.0, 1.0)
    if d < -0.999999:
        # 정반대: a와 직교하는 축 선택
        axis = _cross(a, (1.0, 0.0, 0.0))
        if _dot(axis, axis) < 1e-9:
            axis = _cross(a, (0.0, 0.0, 1.0))
        axis = _norm(axis)
        return (axis[0], axis[1], axis[2], 0.0)
    axis = _cross(a, b)
    s = math.sqrt((1 + d) * 2)
    inv = 1 / s
    return (axis[0] * inv, axis[1] * inv, axis[2] * inv, s * 0.5)


def _quat_to_euler_xyz(q):
    x, y, z, w = q
    # XYZ 순서
    ex = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sy = 2 * (w * y - z * x)
    ey = math.asin(max(-1.0, min(1.0, sy)))
    ez = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return (round(ex, 3), round(ey, 3), round(ez, 3))


def _pt(lm, i):
    p = lm[i]
    if isinstance(p, dict):
        return (float(p.get("x", 0)), float(p.get("y", 0)), float(p.get("z", 0)))
    return (float(p[0]), float(p[1]), float(p[2]))


def _vis(lm, i):
    p = lm[i]
    if isinstance(p, dict):
        return float(p.get("visibility", 0))
    return float(p[3]) if len(p) > 3 else 1.0


def _bone_euler(lm, a_idx, b_idx, ref=DOWN):
    try:
        d = _norm(_v(_pt(lm, a_idx), _pt(lm, b_idx)))
    except (IndexError, TypeError, ValueError):
        return (0.0, 0.0, 0.0)
    return _quat_to_euler_xyz(_quat_from_to(ref, d))


def _root_raw(lm2d):
    """2D 한 프레임 → (hipx, hipy, torso_len, vis). 불가시 None."""
    if not lm2d or len(lm2d) < 29:
        return None
    try:
        hx = (_pt(lm2d, L_HIP)[0] + _pt(lm2d, R_HIP)[0]) / 2
        hy = (_pt(lm2d, L_HIP)[1] + _pt(lm2d, R_HIP)[1]) / 2
        sx = (_pt(lm2d, L_SH)[0] + _pt(lm2d, R_SH)[0]) / 2
        sy = (_pt(lm2d, L_SH)[1] + _pt(lm2d, R_SH)[1]) / 2
        torso = math.sqrt((sx - hx) ** 2 + (sy - hy) ** 2)
        vis = min(_vis(lm2d, L_HIP), _vis(lm2d, R_HIP))
        if torso < 1e-6:
            return None
        return (hx, hy, torso, vis)
    except (IndexError, TypeError, ValueError):
        return None


def attach_root(payload: dict, k: float | None = None, win: int = 5) -> dict:
    """full 페이로드 각 프레임에 root[x,y,z](아바타 단위 상대 오프셋) 주입."""
    if k is None:
        try:
            k = float(os.getenv("ROOT_K", "2.0"))
        except ValueError:
            k = 2.0
    frames = payload.get("frames", [])
    ref = None
    for f in frames:
        r = _root_raw(f.get("poseLandmarks") or [])
        if r is not None and r[3] > 0.2:
            ref = (r[0], r[1], r[2])
            break
    if ref is None:
        for f in frames:
            f["root"] = [0.0, 0.0, 0.0]
        return payload
    buf: list = []
    last = [0.0, 0.0, 0.0]
    for f in frames:
        r = _root_raw(f.get("poseLandmarks") or [])
        if r is None or r[3] < 0.2:
            f["root"] = list(last)  # 검출 실패 프레임은 직전값 유지
            continue
        dx = (r[0] - ref[0]) * k
        dy = -(r[1] - ref[1]) * k  # 영상 y(아래+) → 월드 y(위+)
        dz = (r[2] / ref[2] - 1.0) * k * 0.5  # 커지면(다가오면) +z(카메라 쪽)
        buf.append([dx, dy, dz])
        buf = buf[-win:]
        avg = [sum(c) / len(buf) for c in zip(*buf)]
        f["root"] = [round(v, 4) for v in avg]
        last = avg
    return payload


def frame_to_rig(world_lm: list) -> dict | None:
    """한 프레임 world landmarks(33) → 11개 오일러각. 실패시 None."""
    if not world_lm or len(world_lm) < 29:
        return None
    try:
        hip_c = (( _pt(world_lm, L_HIP)[0] + _pt(world_lm, R_HIP)[0]) / 2,
                 (_pt(world_lm, L_HIP)[1] + _pt(world_lm, R_HIP)[1]) / 2,
                 (_pt(world_lm, L_HIP)[2] + _pt(world_lm, R_HIP)[2]) / 2)
        sh_c = ((_pt(world_lm, L_SH)[0] + _pt(world_lm, R_SH)[0]) / 2,
                (_pt(world_lm, L_SH)[1] + _pt(world_lm, R_SH)[1]) / 2,
                (_pt(world_lm, L_SH)[2] + _pt(world_lm, R_SH)[2]) / 2)
        torso = _norm(_v(hip_c, sh_c))
        # torso는 화면-위쪽(-Y)을 가리키므로 ref를 UP으로
        spine = _quat_to_euler_xyz(_quat_from_to(UP, torso))
        neck = _bone_euler(world_lm, L_SH, NOSE, ref=UP)
        joints = {
            "Spine": list(spine),
            "Chest": list(spine),
            "Neck": list(neck),
            "LeftUpperArm": list(_bone_euler(world_lm, L_SH, L_EL)),
            "LeftLowerArm": list(_bone_euler(world_lm, L_EL, L_WR)),
            "RightUpperArm": list(_bone_euler(world_lm, R_SH, R_EL)),
            "RightLowerArm": list(_bone_euler(world_lm, R_EL, R_WR)),
            "LeftUpperLeg": list(_bone_euler(world_lm, L_HIP, L_KNEE)),
            "LeftLowerLeg": list(_bone_euler(world_lm, L_KNEE, L_ANK)),
            "RightUpperLeg": list(_bone_euler(world_lm, R_HIP, R_KNEE)),
            "RightLowerLeg": list(_bone_euler(world_lm, R_KNEE, R_ANK)),
        }
        vis = round(sum(_vis(world_lm, i) for i in
                        (L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, L_KNEE, R_KNEE)) / 10, 3)
        return {"joints": joints, "vis": vis}
    except (IndexError, TypeError, ValueError):
        return None


def hand_to_joints(hand_lm: list, side: str = "Right") -> dict | None:
    """손 21점 → {wrist, fingers{5×3}, vis}. Kalidokit HandSolver 포팅 사용."""
    if not hand_lm or len(hand_lm) < 21:
        return None
    try:
        from app.services.kalidokit import solve_hand
        solved = solve_hand(hand_lm, side)
        if solved is None:
            return None
        vis = round(sum(_vis(hand_lm, i) for i in (0, 5, 9, 13, 17)) / 5, 3)
        solved["vis"] = vis
        return solved
    except (IndexError, TypeError, ValueError):
        return None


def frame_hand_joints(hands_entries: list) -> dict:
    """프레임의 hands[] → {Left:{...}|None, Right:{...}|None}. 라벨 불명확시 빈 슬롯에."""
    out: dict = {"Left": None, "Right": None}
    for h in hands_entries or []:
        side = str(h.get("side", "")).capitalize()
        if side not in ("Left", "Right"):
            side = "Left" if out["Left"] is None else "Right"
        lms = h.get("world") or h.get("landmarks") or []
        joints = hand_to_joints(lms, side)
        if joints is None:
            continue
        if out[side] is None:
            out[side] = joints
    return out


def attach_hand_joints(payload: dict) -> dict:
    """full 페이로드 각 프레임에 handJoints(손가락 오일러각) 주입. 저장 전에 호출."""
    for f in payload.get("frames", []):
        f["handJoints"] = frame_hand_joints(f.get("hands") or [])
    return payload


def attach_foot_joints(payload: dict) -> dict:
    """full 페이로드 각 프레임에 footJoints(발/발가락) 주입. 저장 전에 호출."""
    for f in payload.get("frames", []):
        f["footJoints"] = foot_to_joints(f.get("poseWorldLandmarks") or [])
    return payload


# carry-forward용 관절별 가시성 기준 랜드마크
JOINT_VIS = {
    "Spine": (23, 24, 11, 12), "Chest": (23, 24, 11, 12),
    "Neck": (11, 12, 0), "Head": (11, 12, 0), "Hips": (23, 24),
    "LeftUpperArm": (11, 13), "LeftLowerArm": (13, 15),
    "RightUpperArm": (12, 14), "RightLowerArm": (14, 16),
    "LeftUpperLeg": (23, 25), "LeftLowerLeg": (25, 27),
    "RightUpperLeg": (24, 26), "RightLowerLeg": (26, 28),
    "LeftFoot": (27, 31), "LeftToe": (29, 31),
    "RightFoot": (28, 32), "RightToe": (30, 32),
}
VIS_HOLD = 0.25


def _joint_ok(world, name):
    try:
        return all(float(world[i].get("visibility", 0)) >= VIS_HOLD
                   for i in JOINT_VIS[name])
    except (IndexError, TypeError, ValueError, AttributeError):
        return False


def wrist_landmarks(hands_entries):
    """FK 손목 방향용 {Left:[21점]|None, Right:...}. world 우선."""
    out = {"Left": None, "Right": None}
    for h in hands_entries or []:
        side = str(h.get("side", "")).capitalize()
        if side not in ("Left", "Right"):
            side = "Left" if out["Left"] is None else "Right"
        lms = h.get("world") or h.get("landmarks") or []
        if len(lms) >= 21 and out[side] is None:
            out[side] = lms
    return out


def attach_fk_joints(payload: dict) -> dict:
    """full 페이로드 각 프레임에 fkJoints(FK 17관절) 주입. 저vis는 직전값 유지."""
    from app.services.fk import solve_fk
    prev: dict = {}
    for f in payload.get("frames", []):
        world = f.get("poseWorldLandmarks") or []
        fk = solve_fk(world, wrist_landmarks(f.get("hands") or [])) or {}
        joints = fk  # solve_fk는 _wrist 채널 포함
        held = {}
        for name in RIG_JOINTS:
            if name in joints and _joint_ok(world, name):
                prev[name] = joints[name]
                held[name] = joints[name]
            elif name in prev:
                held[name] = prev[name]
            else:
                held[name] = [0.0, 0.0, 0.0]
        f["fkJoints"] = held
        # 손목: FK 손목을 handJoints에 반영 (손 검출 없어도 포즈 기반으로 움직임)
        hj = f.get("handJoints") or {"Left": None, "Right": None}
        for side in ("Left", "Right"):
            w = (fk.get("_wrist") or {}).get(side)
            if w is None:
                continue
            if hj.get(side) is None:
                try:
                    pv = float(world[15 if side == "Left" else 16].get("visibility", 0))
                except (IndexError, TypeError, ValueError, AttributeError):
                    pv = 0.0
                if pv < VIS_HOLD and side in prev.get("_wrist", {}):
                    w = prev["_wrist"][side]
                hj[side] = {"wrist": w, "fingers": {}, "vis": round(pv, 3)}
            else:
                hj[side]["wrist"] = w
        f["handJoints"] = hj
        prev["_wrist"] = {s: (hj.get(s) or {}).get("wrist") for s in ("Left", "Right")}
    return payload


def attach_neck_head(payload: dict) -> dict:
    """full 페이로드 각 프레임에 neckJoints {neck, head=neck×0.5} 주입."""
    for f in payload.get("frames", []):
        world = f.get("poseWorldLandmarks") or []
        if world and len(world) >= 29:
            try:
                neck = list(_bone_euler(world, L_SH, NOSE, ref=UP))
            except (IndexError, TypeError, ValueError):
                neck = [0.0, 0.0, 0.0]
        else:
            neck = [0.0, 0.0, 0.0]
        f["neckJoints"] = {"neck": neck,
                           "head": [round(c * 0.5, 3) for c in neck]}
    return payload


def foot_to_joints(world_lm: list) -> dict:
    """발목→발끝·뒤꿈치→발끝 방향 → 발/발가락 오일러. 실패시 빈 dict."""
    out: dict = {}
    try:
        if not world_lm or len(world_lm) < 33:
            return out
        out["LeftFoot"] = list(_bone_euler(world_lm, L_ANK, L_FOOT, ref=FWD))
        out["RightFoot"] = list(_bone_euler(world_lm, R_ANK, R_FOOT, ref=FWD))
        out["LeftToe"] = list(_bone_euler(world_lm, L_HEEL, L_FOOT, ref=FWD))
        out["RightToe"] = list(_bone_euler(world_lm, R_HEEL, R_FOOT, ref=FWD))
    except (IndexError, TypeError, ValueError):
        pass
    return out
