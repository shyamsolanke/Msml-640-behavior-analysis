import math
import cv2
import numpy as np
import torch
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from ultralytics import YOLO

_BYTETRACK_CFG = str(Path(__file__).parent.parent / "config" / "bytetrack.yaml")

WEIGHTS = str(Path(__file__).parent.parent / "weights" / "yolov8m.pt")
CONF = 0.5
IMGSZ = 960
TRAIL_LEN = 30
PERSON_CLASS = 0
GRID = 4
GAMMA = 0.75
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
_GAMMA_TABLE = np.array([
    ((i / 255.0) ** GAMMA) * 255 for i in range(256)
], dtype=np.uint8)


@dataclass
class Track:
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float

    @property
    def cx(self):
        return (self.x1 + self.x2) / 2

    @property
    def cy(self):
        return (self.y1 + self.y2) / 2


class Tracker:
    def __init__(self, weights=WEIGHTS, conf=CONF, imgsz=IMGSZ, trail_len=TRAIL_LEN, step=1):
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.trail_len = trail_len
        self.step = step
        self.trails = defaultdict(list)
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self._frame_idx = 0
        self._last_tracks = []
        self._prev_centroid = {}
        self._prev_frame_id = {}
        self._prev_speed = {}
        self._frames_tracked = defaultdict(int)
        self._dwell_frames = defaultdict(int)
        self._dwell_cell = {}

    def _enhance(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = _CLAHE.apply(l)
        lab = cv2.merge((l, a, b))
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return cv2.LUT(frame, _GAMMA_TABLE)

    def update(self, frame):
        self._frame_idx += 1
        if self.step > 1 and self._frame_idx % self.step != 1:
            return self._last_tracks

        results = self.model.track(
            self._enhance(frame),
            conf=self.conf,
            classes=[PERSON_CLASS],
            device=self.device,
            imgsz=self.imgsz,
            persist=True,
            verbose=False,
            tracker=_BYTETRACK_CFG,
        )
        tracks = []
        r = results[0]
        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.int().cpu().numpy()
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for tid, (x1, y1, x2, y2), c in zip(ids, boxes, confs):
                t = Track(int(tid), float(x1), float(y1), float(x2), float(y2), float(c))
                self.trails[t.track_id].append((t.cx, t.cy))
                if len(self.trails[t.track_id]) > self.trail_len:
                    self.trails[t.track_id].pop(0)
                tracks.append(t)
        self._last_tracks = tracks
        return tracks

    def trail(self, track_id):
        return self.trails.get(track_id, [])

    def build_record(self, frame_id, frame, fps):
        tracks = self.update(frame)
        h, w = frame.shape[:2]
        cell_w = w / GRID
        cell_h = h / GRID

        objects = []
        velocities = []

        for t in tracks:
            tid = t.track_id
            self._frames_tracked[tid] += 1

            prev_cx, prev_cy = self._prev_centroid.get(tid, (t.cx, t.cy))
            prev_fid = self._prev_frame_id.get(tid, frame_id)
            dt = frame_id - prev_fid

            if dt > 0:
                vx = (t.cx - prev_cx) / dt
                vy = (t.cy - prev_cy) / dt
            else:
                vx, vy = 0.0, 0.0

            speed = math.sqrt(vx ** 2 + vy ** 2) * fps
            prev_speed = self._prev_speed.get(tid, speed)
            acceleration = (speed - prev_speed) * fps / max(dt, 1)
            direction_angle = math.degrees(math.atan2(vy, vx))

            col = min(int(t.cx / cell_w), GRID - 1)
            row = min(int(t.cy / cell_h), GRID - 1)
            if self._dwell_cell.get(tid) == (row, col):
                self._dwell_frames[tid] += 1
            else:
                self._dwell_cell[tid] = (row, col)
                self._dwell_frames[tid] = 1

            self._prev_centroid[tid] = (t.cx, t.cy)
            self._prev_frame_id[tid] = frame_id
            self._prev_speed[tid] = speed
            velocities.append((vx * fps, vy * fps))

            objects.append({
                "track_id": tid,
                "bbox": (t.x1, t.y1, t.x2, t.y2),
                "confidence": t.conf,
                "centroid": (t.cx, t.cy),
                "velocity": (round(vx * fps, 3), round(vy * fps, 3)),
                "speed": round(speed, 3),
                "trajectory": list(self.trails[tid]),
                "temporal_features": {
                    "acceleration": round(acceleration, 3),
                    "direction_angle": round(direction_angle, 2),
                    "dwell_time": self._dwell_frames[tid],
                    "frames_tracked": self._frames_tracked[tid],
                },
            })

        n = len(objects)
        if n > 0:
            avg_vx = sum(v[0] for v in velocities) / n
            avg_vy = sum(v[1] for v in velocities) / n
            motion_level = sum(o["speed"] for o in objects) / n
        else:
            avg_vx, avg_vy, motion_level = 0.0, 0.0, 0.0

        return {
            "frame_id": frame_id,
            "timestamp": round(frame_id / fps, 4),
            "image_shape": (h, w, frame.shape[2] if frame.ndim == 3 else 1),
            "objects": objects,
            "global_context": {
                "num_people": n,
                "crowd_density": round(n / ((h * w) / 1e6), 4),
                "scene_motion_level": round(motion_level, 3),
                "global_motion_vector": (round(avg_vx, 3), round(avg_vy, 3)),
            },
        }
