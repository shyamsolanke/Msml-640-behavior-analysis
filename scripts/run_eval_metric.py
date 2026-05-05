import argparse
import configparser
import csv
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import motmetrics as mm
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.tracker import Tracker  # noqa: E402

WEIGHTS = str(Path(__file__).parent.parent / "weights" / "yolov8m.pt")
CONF = 0.25
IMGSZ = 960
IOU_MAX = 0.5

MOT17_ROOT = Path(__file__).parent.parent / "data" / "mot17" / "train"
OUT_ROOT = Path(__file__).parent.parent / "outputs"

TRACKS_SCHEMA = {
    "frame_id":  "int32",
    "track_id":  "int32",
    "bbox_x1":   "float32",
    "bbox_y1":   "float32",
    "bbox_x2":   "float32",
    "bbox_y2":   "float32",
    "bbox_cx":   "float32",
    "bbox_cy":   "float32",
    "mask_cx":   "float32",
    "mask_cy":   "float32",
    "mask_area": "float32",
    "confidence":"float32",
}

log = logging.getLogger("phase2")


def _read_fps(seq_dir):
    ini = seq_dir / "seqinfo.ini"
    if not ini.exists():
        return 30.0
    cfg = configparser.ConfigParser()
    cfg.read(ini)
    try:
        return float(cfg.get("Sequence", "frameRate"))
    except Exception:
        return 30.0


def _parse_gt(gt_path):
    index = {}
    with open(gt_path, newline="") as f:
        for row in csv.reader(f):
            if not row or len(row) < 9:
                continue
            if int(row[7]) != 1 or float(row[6]) != 1.0:
                continue
            frame = int(row[0])
            index.setdefault(frame, []).append({
                "id": int(row[1]),
                "x": float(row[2]), "y": float(row[3]),
                "w": float(row[4]), "h": float(row[5]),
            })
    return index


def _run_tracking(seq_dir):
    img_dir = seq_dir / "img1"
    frame_paths = sorted(img_dir.glob("*.jpg"))
    n = len(frame_paths)
    log.info("tracking %d frames from %s", n, img_dir)

    tracker = Tracker(weights=WEIGHTS, conf=CONF, imgsz=IMGSZ, step=1)
    rows = []

    for i, path in enumerate(frame_paths):
        frame_id = int(path.stem)
        frame = cv2.imread(str(path))
        if frame is None:
            log.warning("could not read %s, skipping", path)
            continue
        tracks = tracker.update(frame)
        for t in tracks:
            cx = (t.x1 + t.x2) / 2.0
            cy = (t.y1 + t.y2) / 2.0
            rows.append({
                "frame_id":  frame_id,
                "track_id":  t.track_id,
                "bbox_x1":   t.x1,
                "bbox_y1":   t.y1,
                "bbox_x2":   t.x2,
                "bbox_y2":   t.y2,
                "bbox_cx":   cx,
                "bbox_cy":   cy,
                "mask_cx":   np.nan,
                "mask_cy":   np.nan,
                "mask_area": np.nan,
                "confidence": t.conf,
            })
        if (i + 1) % 100 == 0:
            log.info("  frame %d / %d", i + 1, n)

    if not rows:
        return pd.DataFrame({c: pd.Series(dtype=t) for c, t in TRACKS_SCHEMA.items()})
    df = pd.DataFrame(rows).astype(TRACKS_SCHEMA)
    return df.sort_values(["frame_id", "track_id"]).reset_index(drop=True)


def _evaluate(df, gt_by_frame, seq_name):
    hyp_by_frame = {}
    for row in df.itertuples(index=False):
        hyp_by_frame.setdefault(int(row.frame_id), []).append({
            "id": int(row.track_id),
            "x": float(row.bbox_x1),
            "y": float(row.bbox_y1),
            "w": float(row.bbox_x2 - row.bbox_x1),
            "h": float(row.bbox_y2 - row.bbox_y1),
        })

    all_frames = sorted(set(gt_by_frame) | set(hyp_by_frame))
    acc = mm.MOTAccumulator(auto_id=True)

    for f in all_frames:
        gts = gt_by_frame.get(f, [])
        hyps = hyp_by_frame.get(f, [])
        gt_ids = [g["id"] for g in gts]
        hyp_ids = [h["id"] for h in hyps]
        if gts and hyps:
            gt_xywh = np.array([[g["x"], g["y"], g["w"], g["h"]] for g in gts], float)
            hyp_xywh = np.array([[h["x"], h["y"], h["w"], h["h"]] for h in hyps], float)
            dists = mm.distances.iou_matrix(gt_xywh, hyp_xywh, max_iou=IOU_MAX)
        else:
            dists = np.empty((len(gts), len(hyps)))
        acc.update(gt_ids, hyp_ids, dists)

    mh = mm.metrics.create()
    metric_names = [
        "idf1", "mota", "motp", "num_switches",
        "mostly_tracked", "mostly_lost",
        "num_false_positives", "num_misses", "num_objects",
    ]
    summary = mh.compute(acc, metrics=metric_names, name=seq_name)
    r = summary.loc[seq_name]

    def _f(v):
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return None
        return float(v) if isinstance(v, (np.floating, float)) else int(v)

    return {
        "seq_name":            seq_name,
        "idf1":                _f(r["idf1"]),
        "mota":                _f(r["mota"]),
        "motp":                _f(r["motp"]),
        "num_switches":        int(r["num_switches"]),
        "mostly_tracked":      int(r["mostly_tracked"]),
        "mostly_lost":         int(r["mostly_lost"]),
        "num_false_positives": int(r["num_false_positives"]),
        "num_misses":          int(r["num_misses"]),
        "num_objects":         int(r["num_objects"]),
        "num_frames_evaluated": len(all_frames),
        "iou_max":             IOU_MAX,
        "weights":             WEIGHTS,
        "conf":                CONF,
        "imgsz":               IMGSZ,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="[phase2] %(message)s")

    p = argparse.ArgumentParser(description="Phase 2 tracking evaluation.")
    p.add_argument("--sequence", required=True, help="MOT17-04-FRCNN or MOT17-09-FRCNN")
    p.add_argument("--mot17-root", default=str(MOT17_ROOT))
    args = p.parse_args()

    seq_dir = Path(args.mot17_root) / args.sequence
    if not seq_dir.exists():
        log.error("sequence directory not found: %s", seq_dir)
        return 1

    fps = _read_fps(seq_dir)
    gt_path = seq_dir / "gt" / "gt.txt"

    out_tracks = OUT_ROOT / "tracks" / f"{args.sequence}_project.parquet"
    out_metrics = OUT_ROOT / "metrics" / f"{args.sequence}_phase2_project.json"
    out_tracks.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)

    log.info("sequence=%s  fps=%.1f  imgsz=%d  conf=%.2f", args.sequence, fps, IMGSZ, CONF)

    t0 = time.perf_counter()
    df = _run_tracking(seq_dir)
    t_track = time.perf_counter() - t0

    df.to_parquet(out_tracks, engine="pyarrow", index=False)
    log.info("tracks saved: %s  rows=%d  (%.1fs)", out_tracks.name, len(df), t_track)

    if gt_path.exists():
        t0 = time.perf_counter()
        gt_by_frame = _parse_gt(gt_path)
        metrics = _evaluate(df, gt_by_frame, args.sequence)
        t_eval = time.perf_counter() - t0

        with open(out_metrics, "w") as f:
            json.dump(metrics, f, indent=2)

        log.info(
            "IDF1=%.3f  MOTA=%.3f  IDs=%d  MT=%d  ML=%d  (%.1fs)",
            metrics["idf1"], metrics["mota"],
            metrics["num_switches"], metrics["mostly_tracked"],
            metrics["mostly_lost"], t_eval,
        )
        log.info("metrics saved: %s", out_metrics.name)

        gate = "PASSED" if metrics["idf1"] >= 0.50 else "FAILED"
        log.info("dev gate (IDF1 >= 0.50): %s  (%.3f)", gate, metrics["idf1"])
    else:
        log.warning("no GT at %s — skipping evaluation", gt_path)

    log.info("done. total %.1fs", t_track)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
