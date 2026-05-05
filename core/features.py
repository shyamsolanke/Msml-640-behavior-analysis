import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

STATIONARY_SPEED = 5.0
PROXIMITY_DIST = 100.0
MIN_DURATION_S = 0.5
MIN_ACTIVE_FRAMES = 5
GRID = 4
HIGH_MOBILITY_FACTOR = 1.5
STATIONARY_THRESHOLD = 0.70
REGION_THRESHOLD = 0.50
SMOOTH_WINDOW = 5
SMOOTH_ORDER = 2


def _smooth(df):
    out = df.copy()
    out["cx_s"] = out["cx"].astype(float)
    out["cy_s"] = out["cy"].astype(float)
    for _, idx in out.groupby("track_id").groups.items():
        sub = out.loc[idx].sort_values("frame_id")
        if len(sub) < SMOOTH_WINDOW:
            continue
        out.loc[sub.index, "cx_s"] = savgol_filter(
            sub["cx"].to_numpy(float), SMOOTH_WINDOW, SMOOTH_ORDER
        )
        out.loc[sub.index, "cy_s"] = savgol_filter(
            sub["cy"].to_numpy(float), SMOOTH_WINDOW, SMOOTH_ORDER
        )
    return out


def _label(features_df):
    out = features_df.copy()
    nonzero = out["avg_speed_px_per_s"][out["avg_speed_px_per_s"] > 0]
    median_speed = float(nonzero.median()) if len(nonzero) else 0.0
    labels = []
    for _, row in out.iterrows():
        if median_speed > 0 and row["avg_speed_px_per_s"] > HIGH_MOBILITY_FACTOR * median_speed:
            labels.append("high_mobility")
        elif row["stationary_ratio"] > STATIONARY_THRESHOLD:
            labels.append("stationary")
        elif row["dominant_cell_pct"] > REGION_THRESHOLD:
            labels.append("region_dominant")
        else:
            labels.append("unlabeled")
    out["label"] = labels
    return out


def compute_features(df, fps, frame_w, frame_h):
    df = _smooth(df)
    cell_w = frame_w / GRID
    cell_h = frame_h / GRID
    rows = []
    for tid, grp in df.groupby("track_id"):
        grp = grp.sort_values("frame_id")
        n = len(grp)
        if n < MIN_ACTIVE_FRAMES:
            continue
        cx = grp["cx_s"].to_numpy(float)
        cy = grp["cy_s"].to_numpy(float)
        frames = grp["frame_id"].to_numpy(int)
        if n >= 2:
            step = np.sqrt(np.diff(cx) ** 2 + np.diff(cy) ** 2)
            total_dist = float(step.sum())
            stationary_ratio = float(np.mean((step * fps) < STATIONARY_SPEED))
        else:
            total_dist = 0.0
            stationary_ratio = 1.0
        active_duration_s = n / fps
        avg_speed = total_dist / active_duration_s if active_duration_s > 0 else 0.0
        col_idx = np.clip(np.floor(cx / cell_w).astype(int), 0, GRID - 1)
        row_idx = np.clip(np.floor(cy / cell_h).astype(int), 0, GRID - 1)
        cell_keys = row_idx * GRID + col_idx
        unique, counts = np.unique(cell_keys, return_counts=True)
        winner = int(unique[np.argmax(counts)])
        dom_pct = float(counts.max()) / n
        rows.append({
            "track_id": int(tid),
            "active_frames": n,
            "first_frame": int(frames.min()),
            "last_frame": int(frames.max()),
            "active_duration_s": round(active_duration_s, 2),
            "total_distance_px": round(total_dist, 1),
            "avg_speed_px_per_s": round(avg_speed, 1),
            "stationary_ratio": round(stationary_ratio, 3),
            "dominant_cell_row": int(winner // GRID),
            "dominant_cell_col": int(winner % GRID),
            "dominant_cell_pct": round(dom_pct, 3),
            "label": "",
        })
    if not rows:
        return pd.DataFrame(columns=[
            "track_id", "active_frames", "first_frame", "last_frame",
            "active_duration_s", "total_distance_px", "avg_speed_px_per_s",
            "stationary_ratio", "dominant_cell_row", "dominant_cell_col",
            "dominant_cell_pct", "label",
        ])
    return _label(pd.DataFrame(rows))


def compute_proximity(df, fps):
    if df.empty:
        return pd.DataFrame(columns=[
            "track_a", "track_b", "start_frame", "end_frame",
            "duration_frames", "duration_s", "min_distance_px",
        ])
    df = _smooth(df)
    cx_mat = df.pivot_table(index="frame_id", columns="track_id", values="cx_s", aggfunc="mean")
    cy_mat = df.pivot_table(index="frame_id", columns="track_id", values="cy_s", aggfunc="mean")
    frame_index = cx_mat.index.to_numpy(int)
    track_ids = list(cx_mat.columns)
    min_frames = max(1, int(round(MIN_DURATION_S * fps)))
    rows = []
    for i in range(len(track_ids)):
        a = track_ids[i]
        cx_a = cx_mat[a].to_numpy(float)
        cy_a = cy_mat[a].to_numpy(float)
        for j in range(i + 1, len(track_ids)):
            b = track_ids[j]
            cx_b = cx_mat[b].to_numpy(float)
            cy_b = cy_mat[b].to_numpy(float)
            d = np.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)
            valid = ~(np.isnan(cx_a) | np.isnan(cx_b) | np.isnan(cy_a) | np.isnan(cy_b))
            close = (d < PROXIMITY_DIST) & valid
            if not close.any():
                continue
            padded = np.concatenate([[False], close, [False]])
            diffs = np.diff(padded.astype(np.int8))
            starts = np.where(diffs == 1)[0]
            ends = np.where(diffs == -1)[0] - 1
            for s, e in zip(starts, ends):
                dur_frames = e - s + 1
                if dur_frames < min_frames:
                    continue
                rows.append({
                    "track_a": int(a),
                    "track_b": int(b),
                    "start_frame": int(frame_index[s]),
                    "end_frame": int(frame_index[e]),
                    "duration_frames": int(dur_frames),
                    "duration_s": round(dur_frames / fps, 2),
                    "min_distance_px": round(float(np.nanmin(d[s: e + 1])), 1),
                })
    if not rows:
        return pd.DataFrame(columns=[
            "track_a", "track_b", "start_frame", "end_frame",
            "duration_frames", "duration_s", "min_distance_px",
        ])
    return pd.DataFrame(rows).sort_values(["track_a", "track_b"]).reset_index(drop=True)


def speed_over_time(df, fps):
    df = _smooth(df)
    result = {}
    for tid, grp in df.groupby("track_id"):
        grp = grp.sort_values("frame_id")
        cx = grp["cx_s"].to_numpy(float)
        cy = grp["cy_s"].to_numpy(float)
        frames = grp["frame_id"].to_numpy(int)
        if len(cx) < 2:
            continue
        speed = np.sqrt(np.diff(cx) ** 2 + np.diff(cy) ** 2) * fps
        result[int(tid)] = pd.DataFrame({"frame_id": frames[1:], "speed_px_s": speed})
    return result
