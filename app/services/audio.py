"""원본 영상 → 오디오(m4a) 추출. ffmpeg는 imageio-ffmpeg 동봉 바이너리 우선.

추출 길이는 실제 사용 프레임(total_frames/fps)으로 잘라 재생 싱크를 맞춤.
ffmpeg 없으면 명확한 설치 안내 에러 (잡은 실패하지 않고 오디오만 스킵).
"""
import os
import subprocess


def _ffmpeg_exe() -> str:
    override = os.getenv("FFMPEG_PATH", "").strip()
    if override and os.path.exists(override):
        return override
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as e:
        raise RuntimeError(
            "오디오 추출에 ffmpeg 필요: pip install imageio-ffmpeg "
            "(또는 FFMPEG_PATH에 실행파일 경로 지정)") from e


def extract_audio(video_path: str, out_path: str, duration_sec: float = 0.0) -> str:
    """video → m4a(aac). duration_sec>0이면 그 길이만큼만. 반환: 경로."""
    exe = _ffmpeg_exe()
    bitrate = os.getenv("AUDIO_BITRATE", "128k")
    cmd = [exe, "-y", "-v", "error", "-i", video_path, "-vn",
           "-c:a", "aac", "-b:a", bitrate]
    if duration_sec and duration_sec > 0:
        cmd += ["-t", f"{duration_sec:.3f}"]
    cmd.append(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"오디오 추출 실패: {(r.stderr or '').strip()[:200]}")
    return out_path
