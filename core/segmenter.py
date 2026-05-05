from __future__ import annotations

import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch

DEFAULT_WEIGHTS = str(Path(__file__).parent.parent / "weights" / "mobile_sam.pt")
DEFAULT_MODEL_TYPE = "vit_t"
DEFAULT_SAMPLE_EVERY_N = 5

_WEIGHTS_URL = (
    "https://github.com/ChaoningZhang/MobileSAM/raw/01ea8d0/weights/mobile_sam.pt"
)


def _ensure_weights(path: str) -> None:
    """Download mobile_sam.pt if not already present."""
    p = Path(path)
    if p.exists():
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    print(f"[segmenter] Downloading MobileSAM weights to {p} ...")
    urllib.request.urlretrieve(_WEIGHTS_URL, p)
    print(f"[segmenter] Download complete ({p.stat().st_size / 1e6:.1f} MB)")


@dataclass
class TrackMask:
    frame_id: int
    track_id: int
    mask_cx: float
    mask_cy: float
    mask_area: float


@dataclass
class FrameMasks:
    frame_id: int
    masks: List[TrackMask] = field(default_factory=list)


def _device():
    return 0 if torch.cuda.is_available() else "cpu"


def _largest_cc(mask_bool: np.ndarray) -> np.ndarray:
    mask_u8 = mask_bool.astype(np.uint8)
    if mask_u8.sum() == 0:
        return mask_u8
    n, comps = cv2.connectedComponents(mask_u8, connectivity=4)
    if n <= 2:
        return mask_u8
    sizes = np.bincount(comps.ravel())
    sizes[0] = 0
    return (comps == int(np.argmax(sizes))).astype(np.uint8)


def _centroid_and_area(mask_u8: np.ndarray) -> Tuple[float, float, float]:
    ys, xs = np.where(mask_u8 > 0)
    if xs.size == 0:
        return float("nan"), float("nan"), 0.0
    return float(xs.mean()), float(ys.mean()), float(xs.size)


def segment_persons(
    frame_iter,
    tracks_df: pd.DataFrame,
    weights: str = DEFAULT_WEIGHTS,
    model_type: str = DEFAULT_MODEL_TYPE,
    sample_every_n: int = DEFAULT_SAMPLE_EVERY_N,
) -> Iterator[FrameMasks]:
    """Yield FrameMasks for every Nth frame (anchored on frame_id == 1).

    Loads MobileSAM once, then for each sampled frame prompts with each
    tracked bbox to get a per-person mask and its centroid/area.
    """
    from mobile_sam import SamPredictor, sam_model_registry

    _ensure_weights(weights)
    sam = sam_model_registry[model_type](checkpoint=weights)
    sam.to(_device()).eval()
    predictor = SamPredictor(sam)

    by_frame = {fid: grp for fid, grp in tracks_df.groupby("frame_id")}

    for frame_id, bgr in frame_iter:
        if (frame_id - 1) % sample_every_n != 0:
            continue
        if frame_id not in by_frame:
            yield FrameMasks(frame_id=frame_id)
            continue

        predictor.set_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        masks_out: List[TrackMask] = []
        for r in by_frame[frame_id].itertuples(index=False):
            box = np.array(
                [r.bbox_x1, r.bbox_y1, r.bbox_x2, r.bbox_y2], dtype=np.float32
            )
            logits, _, _ = predictor.predict(box=box, multimask_output=False)
            mask_u8 = _largest_cc(logits[0] > 0)
            cx, cy, area = _centroid_and_area(mask_u8)
            masks_out.append(TrackMask(
                frame_id=int(frame_id),
                track_id=int(r.track_id),
                mask_cx=cx,
                mask_cy=cy,
                mask_area=area,
            ))
        yield FrameMasks(frame_id=int(frame_id), masks=masks_out)


def interpolate_mask_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Linear interpolation of mask_cx / mask_cy / mask_area per track."""
    out = df.copy()
    cols = ["mask_cx", "mask_cy", "mask_area"]
    for _, sub in out.groupby("track_id"):
        sub_s = sub.sort_values("frame_id")
        if sub_s[cols].isna().all().all():
            continue
        for c in cols:
            s = sub_s.set_index("frame_id")[c]
            filled = s.interpolate(method="index", limit_direction="both")
            out.loc[sub_s.index, c] = filled.values
    out[cols] = out[cols].astype("float32")
    return out
