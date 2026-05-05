import shutil
import subprocess

import cv2


def _rotated(frame, angle):
    if angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def get_info(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
    cap.release()
    if rotation in (90, 270):
        w, h = h, w
    return fps, w, h, n


def iter_frames(path):
    cap = cv2.VideoCapture(str(path))
    rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield _rotated(frame, rotation)
    cap.release()


def make_writer(path, fps, w, h):
    for codec in ("avc1", "H264", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
        if writer.isOpened():
            return writer
    raise RuntimeError("No supported video codec found")


def to_browser_mp4(src, dst):
    # OpenCV's bundled mp4 encoder uses mp4v, which HTML5 <video> can't play.
    # Re-encode to H.264 + faststart so Streamlit's player works in-browser.
    if not shutil.which("ffmpeg"):
        return False
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "veryfast", "-movflags", "+faststart",
        "-an",
        str(dst),
    ]
    return subprocess.run(cmd, check=False).returncode == 0
