import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from core.tracker import Tracker
from core.overlay import draw
from core.video import get_info, iter_frames, make_writer, to_browser_mp4
from core.features import compute_features, compute_proximity, speed_over_time
from core.summaries import render_summary
from scripts.llm_summaries import render_llm_summary

LABEL_COLORS = {
    "high_mobility":  "#e74c3c",
    "stationary":     "#2ecc71",
    "region_dominant":"#3498db",
    "unlabeled":      "#95a5a6",
}
DATA_DIR = Path(__file__).parent / "data"
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
STEP = 2


def _find_local_videos():
    if not DATA_DIR.exists():
        return []
    return sorted(p for p in DATA_DIR.rglob("*") if p.suffix.lower() in VIDEO_EXTS)


def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("#f8f9fa")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=9, labelpad=6)
    ax.set_ylabel(ylabel, fontsize=9, labelpad=6)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25, linewidth=0.8, linestyle="--")


def _display_summary(summary_text):
    for line in summary_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("Video:"):
            st.markdown(
                f'<div style="font-weight:600;font-size:0.95rem;'
                f'padding:8px 12px;border-radius:6px;'
                f'background:#1e1e2e;color:#cdd6f4;margin-bottom:6px">'
                f'{line}</div>',
                unsafe_allow_html=True,
            )
        elif "high mobility" in line.lower():
            st.markdown(
                f'<div style="padding:7px 12px;border-left:3px solid #e74c3c;'
                f'background:#fdf2f2;border-radius:0 6px 6px 0;'
                f'margin:3px 0;font-size:0.88rem">{line}</div>',
                unsafe_allow_html=True,
            )
        elif "stationary" in line.lower():
            st.markdown(
                f'<div style="padding:7px 12px;border-left:3px solid #2ecc71;'
                f'background:#f2fdf5;border-radius:0 6px 6px 0;'
                f'margin:3px 0;font-size:0.88rem">{line}</div>',
                unsafe_allow_html=True,
            )
        elif "grid cell" in line.lower():
            st.markdown(
                f'<div style="padding:7px 12px;border-left:3px solid #3498db;'
                f'background:#f2f8fd;border-radius:0 6px 6px 0;'
                f'margin:3px 0;font-size:0.88rem">{line}</div>',
                unsafe_allow_html=True,
            )
        elif "close" in line.lower() or "persons" in line.lower():
            st.markdown(
                f'<div style="padding:7px 12px;border-left:3px solid #f39c12;'
                f'background:#fdf9f2;border-radius:0 6px 6px 0;'
                f'margin:3px 0;font-size:0.88rem">{line}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="padding:7px 12px;border-left:3px solid #95a5a6;'
                f'background:#f8f9fa;border-radius:0 6px 6px 0;'
                f'margin:3px 0;font-size:0.88rem">{line}</div>',
                unsafe_allow_html=True,
            )


st.set_page_config(page_title="Person Tracker", layout="wide")

st.markdown(
    "<h1 style='margin-bottom:0'>Person Tracker</h1>"
    "<p style='color:#888;margin-top:4px'>Multi-object tracking with persistent IDs, "
    "motion trails, and behavioral analysis</p><hr style='margin:12px 0 20px'>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## Configuration")
    st.divider()

    local_videos = _find_local_videos()
    source = st.radio("Video source", ["Local (data/)", "Upload"], horizontal=False)

    in_path_direct = None
    uploaded = None

    if source == "Local (data/)":
        if not local_videos:
            st.warning("No videos found in `data/`.")
        else:
            labels = [str(p.relative_to(DATA_DIR.parent)) for p in local_videos]
            choice = st.selectbox("Select video", labels)
            in_path_direct = DATA_DIR.parent / choice
    else:
        uploaded = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"])

    st.divider()
    st.markdown("### Processing")
    ready = in_path_direct is not None or uploaded is not None

    if ready:
        if in_path_direct:
            file_key = f"{in_path_direct.name}_{in_path_direct.stat().st_size}"
            seq_name = in_path_direct.stem
            _tmp_process_path = in_path_direct
        else:
            file_key = f"{uploaded.name}_{uploaded.size}"
            seq_name = Path(uploaded.name).stem
            # Write uploaded bytes to a named temp file so it can be probed/processed
            if st.session_state.get("upload_file_key") != file_key:
                suffix = Path(uploaded.name).suffix
                tmp_upload = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp_upload.write(uploaded.getvalue())
                tmp_upload.close()
                st.session_state.upload_tmp_path = tmp_upload.name
                st.session_state.upload_file_key = file_key
            _tmp_process_path = Path(st.session_state.upload_tmp_path)

        if st.session_state.get("file_key") != file_key:
            st.session_state.file_key = file_key
            st.session_state.video_bytes = None
            st.session_state.features_df = None
            st.session_state.proximity_df = None
            st.session_state.raw_tracks = None
            st.session_state.summary = None
            st.session_state.llm_summary = None
            st.session_state.fps = None
            st.session_state.video_info = None

        if st.session_state.video_bytes is None:
            probe_path = _tmp_process_path
            fps, w, h, n_frames = get_info(probe_path)
            st.session_state.video_info = (fps, w, h, n_frames)
            st.caption(f"{n_frames} frames · {w}×{h} · {fps:.1f} fps")

            max_frames = st.slider(
                "Frame limit (0 = all)",
                min_value=0, max_value=n_frames, value=0, step=10,
            )
            limit = max_frames if max_frames > 0 else n_frames
            st.session_state._limit = limit

            has_openai_key = bool(os.environ.get("OPENAI_API_KEY"))
            llm_enabled = st.checkbox(
                "Generate AI summary (LLM)",
                value=False,
                disabled=not has_openai_key,
                help=(
                    "Calls OpenAI gpt-4o-mini to write a free-text behavioral "
                    "summary. Adds API cost and a few seconds of latency."
                    if has_openai_key
                    else "Set OPENAI_API_KEY to enable."
                ),
            )
            st.session_state._llm_enabled = llm_enabled

            run = st.button("Start Processing", use_container_width=True, type="primary")
            if not run:
                st.stop()

            with tempfile.TemporaryDirectory() as tmp_str:
                tmp = Path(tmp_str)
                out_path = tmp / "output.mp4"
                process_path = probe_path

                tracker = Tracker(step=STEP)
                writer = make_writer(out_path, fps, w, h)
                bar = st.progress(0, text="Initialising...")
                records = []

                for i, frame in enumerate(iter_frames(process_path)):
                    if i >= limit:
                        break
                    frame_id = i + 1
                    rec = tracker.build_record(frame_id, frame, fps)
                    for obj in rec["objects"]:
                        tf = obj["temporal_features"]
                        records.append({
                            "frame_id": frame_id,
                            "track_id": obj["track_id"],
                            "cx": obj["centroid"][0],
                            "cy": obj["centroid"][1],
                            "speed": obj["speed"],
                            "vx": obj["velocity"][0],
                            "vy": obj["velocity"][1],
                            "acceleration": tf["acceleration"],
                            "direction_angle": tf["direction_angle"],
                            "dwell_time": tf["dwell_time"],
                        })
                    writer.write(draw(frame, tracker._last_tracks, tracker.trail))
                    bar.progress((i + 1) / limit, text=f"Frame {i + 1} / {limit}")

                writer.release()
                bar.progress(1.0, text="Done")

                raw_df = pd.DataFrame(records)
                features_df = compute_features(raw_df, fps, w, h)
                proximity_df = compute_proximity(raw_df, fps)

                # Transcode to H.264 so HTML5 <video> can play it in the browser.
                browser_path = tmp / "output_h264.mp4"
                playable = to_browser_mp4(out_path, browser_path) and browser_path.exists()
                st.session_state.video_bytes = (browser_path if playable else out_path).read_bytes()
                st.session_state.raw_tracks = raw_df
                st.session_state.fps = fps
                st.session_state.frame_size = (w, h)
                st.session_state.features_df = features_df
                st.session_state.proximity_df = proximity_df
                st.session_state.summary = render_summary(features_df, proximity_df, seq_name=seq_name)

                if st.session_state.get("_llm_enabled"):
                    with st.spinner("Generating AI summary..."):
                        try:
                            st.session_state.llm_summary = render_llm_summary(
                                features_df, proximity_df,
                                seq_name=seq_name, fps=fps,
                                frame_w=w, frame_h=h,
                            )
                        except Exception as exc:
                            st.session_state.llm_summary = f"_AI summary unavailable: {exc}_"

        else:
            fps_info = st.session_state.video_info
            if fps_info:
                fps_v, w_v, h_v, nf_v = fps_info
                st.caption(f"{nf_v} frames · {w_v}×{h_v} · {fps_v:.1f} fps")
            st.success("Processed")

    st.divider()
    st.markdown(
        "<small style='color:#888'>MSML 640 · Group 6<br>YOLOv8m + ByteTrack</small>",
        unsafe_allow_html=True,
    )


if st.session_state.get("video_bytes") is None:
    st.markdown(
        "<div style='text-align:center;padding:80px 0;color:#888'>"
        "<h3>Select or upload a video to get started</h3>"
        "<p>Supports MP4, MOV, AVI, MKV</p></div>",
        unsafe_allow_html=True,
    )
    st.stop()

features_df = st.session_state.features_df
proximity_df = st.session_state.proximity_df
raw_df = st.session_state.raw_tracks
fps = st.session_state.fps
w, h = st.session_state.frame_size

col_video, col_summary = st.columns([6, 5])

with col_video:
    st.markdown("#### Tracked Video")
    st.video(st.session_state.video_bytes)
    st.download_button(
        "Download tracked video",
        st.session_state.video_bytes,
        "tracked.mp4",
        "video/mp4",
        use_container_width=True,
    )

with col_summary:
    st.markdown("#### Behavioral Summary")
    with st.container(height=260, border=True):
        _display_summary(st.session_state.summary)

    st.markdown("#### AI Summary")
    with st.container(height=260, border=True):
        llm_text = st.session_state.get("llm_summary")
        if llm_text:
            st.markdown(llm_text)
        elif st.session_state.get("_llm_enabled"):
            st.caption("AI summary will appear here after processing.")
        else:
            st.caption(
                "Enable **Generate AI summary (LLM)** in the sidebar before "
                "processing to populate this panel. Requires `OPENAI_API_KEY`."
            )

st.divider()

if features_df.empty:
    st.warning("No tracks had enough frames to compute features.")
    st.stop()

label_counts = features_df["label"].value_counts().to_dict()
n_tracks = len(features_df)
avg_speed = features_df["avg_speed_px_per_s"].mean()

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Tracks", n_tracks)
m2.metric("Avg Speed (px/s)", f"{avg_speed:.1f}")
m3.metric("High Mobility", label_counts.get("high_mobility", 0))
m4.metric("Stationary", label_counts.get("stationary", 0))
m5.metric("Region Dominant", label_counts.get("region_dominant", 0))

st.markdown(
    " ".join(
        f'<span style="display:inline-block;padding:3px 10px;border-radius:12px;'
        f'background:{c}22;border:1px solid {c};color:{c};'
        f'font-size:0.78rem;margin:2px">{l.replace("_"," ")}</span>'
        for l, c in LABEL_COLORS.items()
    ),
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

tab_feat, tab_speed, tab_traj, tab_prox, tab_raw = st.tabs([
    "Features", "Speed", "Trajectories", "Proximity", "Frame Data"
])

with tab_feat:
    st.dataframe(
        features_df.style.apply(
            lambda col: [
                f"background-color:{LABEL_COLORS.get(v,'')}22"
                for v in features_df["label"]
            ] if col.name == "label" else [""] * len(col),
            axis=0,
        ),
        use_container_width=True,
        hide_index=True,
    )

with tab_speed:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Average speed per track**")
        sorted_feat = features_df.sort_values("avg_speed_px_per_s", ascending=True)
        fig, ax = plt.subplots(figsize=(5, max(3, len(sorted_feat) * 0.32)))
        fig.patch.set_facecolor("#ffffff")
        bar_colors = [LABEL_COLORS.get(l, "#95a5a6") for l in sorted_feat["label"]]
        bars = ax.barh(sorted_feat["track_id"].astype(str), sorted_feat["avg_speed_px_per_s"],
                       color=bar_colors, edgecolor="white", linewidth=0.5)
        _style_ax(ax, xlabel="avg speed (px/s)", ylabel="track id")
        ax.legend(
            handles=[mpatches.Patch(color=c, label=l.replace("_", " "))
                     for l, c in LABEL_COLORS.items()],
            fontsize=7, loc="lower right",
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        st.markdown("**Speed over time**")
        sot = speed_over_time(raw_df, fps)
        track_options = sorted(sot.keys())
        if track_options:
            selected = st.selectbox("Track ID", track_options, format_func=lambda x: f"#{x}")
            fig2, ax2 = plt.subplots(figsize=(5, 3))
            fig2.patch.set_facecolor("#ffffff")
            s = sot[selected]
            track_label = (
                features_df.loc[features_df["track_id"] == selected, "label"].values[0]
                if selected in features_df["track_id"].values else "unlabeled"
            )
            ax2.fill_between(s["frame_id"], s["speed_px_s"],
                             alpha=0.15, color=LABEL_COLORS.get(track_label, "#3498db"))
            ax2.plot(s["frame_id"], s["speed_px_s"], linewidth=1.5,
                     color=LABEL_COLORS.get(track_label, "#3498db"))
            ax2.axhline(5.0, color="#888", linestyle="--", linewidth=0.9, label="stationary (5 px/s)")
            _style_ax(ax2, title=f"Track #{selected} — {track_label.replace('_',' ')}",
                      xlabel="frame", ylabel="speed (px/s)")
            ax2.legend(fontsize=7)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

with tab_traj:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Trajectory paths**")
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#f0f2f6")
        unique_ids = raw_df["track_id"].unique()
        cmap = plt.colormaps["tab20"].resampled(max(len(unique_ids), 1))
        for idx, tid in enumerate(sorted(unique_ids)):
            grp = raw_df[raw_df["track_id"] == tid].sort_values("frame_id")
            color = cmap(idx)
            ax.plot(grp["cx"], grp["cy"], linewidth=1.2, alpha=0.8, color=color)
            ax.scatter(grp["cx"].iloc[0], grp["cy"].iloc[0], s=30, zorder=5,
                       color=color, edgecolors="white", linewidth=0.5)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("x (px)", fontsize=9)
        ax.set_ylabel("y (px)", fontsize=9)
        ax.set_title("All trajectories", fontsize=11, fontweight="bold")
        if len(unique_ids) <= 20:
            ax.legend(
                [plt.Line2D([0], [0], color=cmap(i), linewidth=2) for i in range(len(unique_ids))],
                [f"#{tid}" for tid in sorted(unique_ids)],
                fontsize=6, ncol=4, loc="upper right",
            )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        st.markdown("**Distance travelled per track**")
        dist_df = features_df[["track_id", "total_distance_px", "label"]].sort_values(
            "total_distance_px", ascending=True
        )
        fig3, ax3 = plt.subplots(figsize=(5, max(3, len(dist_df) * 0.32)))
        fig3.patch.set_facecolor("#ffffff")
        bar_colors = [LABEL_COLORS.get(l, "#95a5a6") for l in dist_df["label"]]
        ax3.barh(dist_df["track_id"].astype(str), dist_df["total_distance_px"],
                 color=bar_colors, edgecolor="white", linewidth=0.5)
        _style_ax(ax3, xlabel="total distance (px)", ylabel="track id")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

with tab_prox:
    n_events = len(proximity_df)
    if proximity_df.empty:
        st.info("No proximity events detected (threshold: distance < 100 px for ≥ 0.5 s).")
    else:
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown(f"**{n_events} event(s)** — centroid distance < 100 px for ≥ 0.5 s")
            st.dataframe(proximity_df, use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**Duration distribution**")
            fig4, ax4 = plt.subplots(figsize=(4, 3))
            fig4.patch.set_facecolor("#ffffff")
            ax4.hist(proximity_df["duration_s"], bins=min(15, n_events),
                     color="#3498db", edgecolor="white", linewidth=0.8, alpha=0.85)
            _style_ax(ax4, title="Event durations", xlabel="duration (s)", ylabel="count")
            ax4.grid(axis="y", alpha=0.25, linewidth=0.8, linestyle="--")
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close(fig4)

with tab_raw:
    show_cols = [
        "frame_id", "track_id", "cx", "cy",
        "speed", "vx", "vy", "acceleration", "direction_angle", "dwell_time",
    ]
    display_df = raw_df[[c for c in show_cols if c in raw_df.columns]]
    total = len(display_df)
    st.caption(f"{total:,} rows · {display_df['track_id'].nunique()} tracks · "
               f"{display_df['frame_id'].nunique()} frames")
    st.dataframe(display_df, use_container_width=True, hide_index=True)
