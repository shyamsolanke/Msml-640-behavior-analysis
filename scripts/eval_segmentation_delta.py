from __future__ import annotations

import argparse
import configparser
import json
import logging
import time
from pathlib import Path

import cv2
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.segmenter import segment_persons, interpolate_mask_columns  # noqa: E402
from core.features import compute_features                             # noqa: E402

WEIGHTS = str(Path(__file__).parent.parent / "weights" / "mobile_sam.pt")
MOT17_ROOT = Path(__file__).parent.parent / "data" / "mot17" / "train"
OUT_ROOT = Path(__file__).parent.parent / "outputs"
SAMPLE_EVERY_N = 5

log = logging.getLogger("phase4")


def _read_fps(seq_dir: Path) -> float:
    ini = seq_dir / "seqinfo.ini"
    if not ini.exists():
        return 30.0
    cfg = configparser.ConfigParser()
    cfg.read(ini)
    try:
        return float(cfg.get("Sequence", "frameRate"))
    except Exception:
        return 30.0


def _frame_iter(img_dir: Path):
    """Yield (frame_id, bgr_frame) for every jpg in img_dir, sorted."""
    for path in sorted(img_dir.glob("*.jpg")):
        frame_id = int(path.stem)
        frame = cv2.imread(str(path))
        if frame is not None:
            yield frame_id, frame


def _features_from(df: pd.DataFrame, cx_col: str, cy_col: str,
                   fps: float, frame_w: int, frame_h: int) -> pd.DataFrame:
    work = df[["frame_id", "track_id"]].copy()
    work["cx"] = df[cx_col].astype(float)
    work["cy"] = df[cy_col].astype(float)
    work = work.dropna(subset=["cx", "cy"])
    return compute_features(work, fps, frame_w, frame_h)


def _stat(series: pd.Series) -> dict:
    s = series.dropna()
    if len(s) == 0:
        return {"mean": None, "median": None, "std": None, "n": 0}
    return {
        "mean":   round(float(s.mean()), 4),
        "median": round(float(s.median()), 4),
        "std":    round(float(s.std()), 4),
        "n":      int(len(s)),
    }


def main():
    logging.basicConfig(level=logging.INFO, format="[phase4] %(message)s")

    p = argparse.ArgumentParser(description="Phase 4 MobileSAM segmentation.")
    p.add_argument("--sequence", required=True,
                   help="e.g. MOT17-04-FRCNN or MOT17-09-FRCNN")
    p.add_argument("--sample-every-n", type=int, default=SAMPLE_EVERY_N)
    p.add_argument("--mot17-root", default=str(MOT17_ROOT))
    args = p.parse_args()

    seq = args.sequence
    seq_dir = Path(args.mot17_root) / seq
    img_dir = seq_dir / "img1"
    in_tracks = OUT_ROOT / "tracks" / f"{seq}_project.parquet"
    out_delta = OUT_ROOT / "metrics" / f"{seq}_phase4_project.json"

    if not seq_dir.exists():
        log.error("sequence directory not found: %s", seq_dir)
        return 1
    if not in_tracks.exists():
        log.error("tracks not found: %s  (run run_eval_metric.py first)", in_tracks)
        return 1

    fps = _read_fps(seq_dir)
    first_jpg = sorted(img_dir.glob("*.jpg"))[0]
    img0 = cv2.imread(str(first_jpg))
    frame_h, frame_w = img0.shape[:2]

    df = pd.read_parquet(in_tracks, engine="pyarrow")
    log.info("sequence=%s  fps=%.1f  size=%dx%d  tracks=%d  rows=%d",
             seq, fps, frame_w, frame_h, df["track_id"].nunique(), len(df))

    # ---- Stage 1: MobileSAM segmentation -----------------------------------
    t0 = time.perf_counter()
    sampled = []
    for fm in segment_persons(
        _frame_iter(img_dir), df,
        weights=WEIGHTS,
        sample_every_n=args.sample_every_n,
    ):
        for tm in fm.masks:
            sampled.append({
                "frame_id":  int(tm.frame_id),
                "track_id":  int(tm.track_id),
                "mask_cx":   float(tm.mask_cx),
                "mask_cy":   float(tm.mask_cy),
                "mask_area": float(tm.mask_area),
            })
    t_seg = time.perf_counter() - t0
    log.info("segmentation: %d detections  (%.1fs)", len(sampled), t_seg)

    # ---- Stage 2: merge mask results into tracks DataFrame -----------------
    t0 = time.perf_counter()
    if sampled:
        sm_df = pd.DataFrame(sampled)
        merged = df.merge(sm_df, on=["frame_id", "track_id"],
                          how="left", suffixes=("", "_new"))
        for c in ("mask_cx", "mask_cy", "mask_area"):
            nc = f"{c}_new"
            if nc in merged.columns:
                merged[c] = (
                    merged[nc].astype("float32")
                    .combine_first(merged[c].astype("float32"))
                )
                merged = merged.drop(columns=[nc])
        df = merged

    df = interpolate_mask_columns(df)
    t_merge = time.perf_counter() - t0
    log.info("merge + interpolate  (%.2fs)", t_merge)

    # ---- Stage 3: features using bbox centroid vs mask centroid ------------
    t0 = time.perf_counter()
    feats_bbox = _features_from(df, "bbox_cx", "bbox_cy", fps, frame_w, frame_h)
    feats_mask = _features_from(df, "mask_cx", "mask_cy", fps, frame_w, frame_h)
    t_feat = time.perf_counter() - t0
    log.info("features: bbox=%d tracks  mask=%d tracks  (%.2fs)",
             len(feats_bbox), len(feats_mask), t_feat)

    # ---- Stage 4: save delta JSON ------------------------------------------
    delta = {
        "seq_name": seq,
        "config": {
            "sample_every_n": args.sample_every_n,
            "model_type":     "vit_t",
            "weights":        WEIGHTS,
            "fps":            fps,
            "frame_w":        frame_w,
            "frame_h":        frame_h,
        },
        "n_tracks_bbox":    int(len(feats_bbox)),
        "n_tracks_mask":    int(len(feats_mask)),
        "avg_speed_px_per_s": {
            "bbox": _stat(feats_bbox["avg_speed_px_per_s"]),
            "mask": _stat(feats_mask["avg_speed_px_per_s"]),
        },
        "stationary_ratio": {
            "bbox": _stat(feats_bbox["stationary_ratio"]),
            "mask": _stat(feats_mask["stationary_ratio"]),
        },
        "dominant_cell_pct": {
            "bbox": _stat(feats_bbox["dominant_cell_pct"]),
            "mask": _stat(feats_mask["dominant_cell_pct"]),
        },
        "label_counts_bbox": feats_bbox["label"].value_counts().to_dict(),
        "label_counts_mask": feats_mask["label"].value_counts().to_dict(),
        "runtime_s": {
            "segmentation": round(t_seg, 2),
            "merge_interp": round(t_merge, 2),
            "features":     round(t_feat, 2),
        },
    }

    out_delta.parent.mkdir(parents=True, exist_ok=True)
    with open(out_delta, "w") as f:
        json.dump(delta, f, indent=2)
    log.info("delta saved: %s", out_delta.name)

    speed_delta = (
        feats_mask["avg_speed_px_per_s"].mean()
        - feats_bbox["avg_speed_px_per_s"].mean()
    )
    log.info("speed delta (mask - bbox): %+.2f px/s", speed_delta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
