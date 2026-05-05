"""LLM-based behavioral summaries using OpenAI Chat Completions.

"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd


SYSTEM_PROMPT = (
    "You are analyzing pedestrian-tracking data from a single video clip. "
    "You receive per-person motion features (speed, stationary ratio, "
    "dominant grid cell, active duration) and pairwise proximity events "
    "(when two people were close for at least 0.5s). "
    "Write a concise behavioral summary (5-10 sentences) describing what "
    "people are doing, who interacts with whom, and any notable patterns "
    "(loitering, fast movers, group formations, crowd flow). Cite specific "
    "track IDs, durations, and speeds. Do NOT invent data not present in "
    "the input. If the data is sparse, say so."
)


def _build_payload(
    features_df: pd.DataFrame,
    proximity_df: Optional[pd.DataFrame],
    *,
    seq_name: str,
    fps: float,
    frame_w: int,
    frame_h: int,
) -> dict:
    return {
        "sequence": seq_name,
        "fps": float(fps),
        "frame_size": [int(frame_w), int(frame_h)],
        "n_tracks": int(len(features_df)),
        "tracks": features_df.to_dict(orient="records") if len(features_df) else [],
        "proximity_events": (
            proximity_df.to_dict(orient="records")
            if proximity_df is not None and len(proximity_df) else []
        ),
    }


def render_llm_summary(
    features_df: pd.DataFrame,
    proximity_df: Optional[pd.DataFrame],
    *,
    seq_name: str,
    fps: float,
    frame_w: int,
    frame_h: int,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
) -> str:
    """Call OpenAI Chat Completions with per-track features as the payload.

    Returns the assistant message text. Raises RuntimeError if openai is
    not installed or OPENAI_API_KEY is not set.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set. "
            "Set it before requesting an LLM summary."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The `openai` package is not installed. "
            "Run `pip install openai` and retry."
        ) from exc

    if features_df is None or len(features_df) == 0:
        return (
            f"Video: {seq_name}. No tracks were observed, "
            "so there is no behavior to describe."
        )

    payload = _build_payload(
        features_df, proximity_df,
        seq_name=seq_name, fps=fps, frame_w=frame_w, frame_h=frame_h,
    )

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, default=str)},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="LLM behavioral summary CLI.")
    p.add_argument("--features", dest="features_path", required=True,
                   help="Path to features parquet or CSV.")
    p.add_argument("--proximity", dest="proximity_path", default=None)
    p.add_argument("--seq", dest="seq_name", required=True)
    p.add_argument("--fps", type=float, required=True)
    p.add_argument("--frame-w", type=int, required=True)
    p.add_argument("--frame-h", type=int, required=True)
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--out", dest="out_path", required=True)
    args = p.parse_args()

    fp = Path(args.features_path)
    feats = pd.read_parquet(fp) if fp.suffix == ".parquet" else pd.read_csv(fp)

    prox = None
    if args.proximity_path:
        pp = Path(args.proximity_path)
        prox = pd.read_parquet(pp) if pp.suffix == ".parquet" else pd.read_csv(pp)

    text = render_llm_summary(
        feats, prox,
        seq_name=args.seq_name,
        fps=args.fps,
        frame_w=args.frame_w,
        frame_h=args.frame_h,
        model=args.model,
    )

    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"[llm_summaries] -> {args.out_path}  ({len(text)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
