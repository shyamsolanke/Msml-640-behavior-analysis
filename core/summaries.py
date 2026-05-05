import pandas as pd

PROXIMITY_THRESHOLD = 100.0
MAX_TRACKS = 5
MAX_PAIRS = 3

HEADER = (
    "Video: {seq}. {n_tracks} person track(s) observed; "
    "median speed {median_speed:.1f} px/s."
)

TRACK_TEMPLATES = {
    "high_mobility": (
        "Person #{track_id} moved at {avg_speed:.1f} px/s on average over "
        "{duration:.1f} s — high mobility."
    ),
    "stationary": (
        "Person #{track_id} remained stationary for {stationary_ratio:.0%} "
        "of the time they were visible."
    ),
    "region_dominant": (
        "Person #{track_id} stayed mostly in grid cell ({row}, {col}) "
        "({pct:.0%} of their visible frames)."
    ),
    "unlabeled": (
        "Person #{track_id} traveled {distance:.0f} px over {duration:.1f} s."
    ),
}

PAIR_TEMPLATE = (
    "Persons #{a} and #{b} were close (< {thresh:.0f} px) for "
    "{n_events} interaction event(s) totaling {total_s:.1f} s."
)


def _track_score(row):
    label_bonus = 2.0 if row["label"] not in ("unlabeled", "") else 0.0
    return (
        label_bonus
        + float(row["dominant_cell_pct"])
        + min(float(row["avg_speed_px_per_s"]), 200.0) / 200.0
    )


def _track_sentence(row):
    tid = int(row["track_id"])
    label = row["label"]
    if label == "high_mobility":
        return TRACK_TEMPLATES["high_mobility"].format(
            track_id=tid,
            avg_speed=float(row["avg_speed_px_per_s"]),
            duration=float(row["active_duration_s"]),
        )
    if label == "stationary":
        return TRACK_TEMPLATES["stationary"].format(
            track_id=tid,
            stationary_ratio=float(row["stationary_ratio"]),
        )
    if label == "region_dominant":
        return TRACK_TEMPLATES["region_dominant"].format(
            track_id=tid,
            row=int(row["dominant_cell_row"]),
            col=int(row["dominant_cell_col"]),
            pct=float(row["dominant_cell_pct"]),
        )
    return TRACK_TEMPLATES["unlabeled"].format(
        track_id=tid,
        distance=float(row["total_distance_px"]),
        duration=float(row["active_duration_s"]),
    )


def render_summary(features_df, proximity_df, seq_name="video"):
    n_tracks = len(features_df)
    if n_tracks == 0:
        return f"Video: {seq_name}. No tracks observed.\n"

    nonzero = features_df["avg_speed_px_per_s"][features_df["avg_speed_px_per_s"] > 0]
    median_speed = float(nonzero.median()) if len(nonzero) else 0.0

    header = HEADER.format(seq=seq_name, n_tracks=n_tracks, median_speed=median_speed)

    scored = features_df.copy()
    scored["_score"] = scored.apply(_track_score, axis=1)
    top = scored.sort_values(["_score", "track_id"], ascending=[False, True]).head(MAX_TRACKS)
    track_lines = [_track_sentence(r) for _, r in top.iterrows()]

    pair_lines = []
    if proximity_df is not None and not proximity_df.empty:
        agg = (
            proximity_df.groupby(["track_a", "track_b"], sort=True)
            .agg(
                event_count=("duration_frames", "size"),
                total_close_s=("duration_s", "sum"),
                min_distance_px=("min_distance_px", "min"),
            )
            .reset_index()
        )
        agg["score"] = agg["event_count"].astype(float) + agg["total_close_s"].astype(float) / 5.0
        agg = agg.sort_values(["score", "track_a", "track_b"], ascending=[False, True, True]).head(MAX_PAIRS)
        for _, p in agg.iterrows():
            pair_lines.append(
                PAIR_TEMPLATE.format(
                    a=int(p["track_a"]),
                    b=int(p["track_b"]),
                    thresh=PROXIMITY_THRESHOLD,
                    n_events=int(p["event_count"]),
                    total_s=float(p["total_close_s"]),
                )
            )

    parts = [header, ""] + track_lines
    if pair_lines:
        parts += [""] + pair_lines
    return "\n".join(parts) + "\n"
