from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("human_eval")

DEFAULT_EVALUATORS = ["E1", "E2", "E3", "E4", "E5"]
SEED = 20260426

STIMULI: List[Tuple[str, str]] = [
    ("MOT17-04-FRCNN", "behavior"),
    ("MOT17-04-FRCNN", "counts"),
    ("MOT17-09-FRCNN", "behavior"),
    ("MOT17-09-FRCNN", "counts"),
    ("demo_clip",      "behavior"),
    ("demo_clip",      "counts"),
]

QUESTIONS = ["q1_accuracy", "q2_usefulness"]

OUT_ROOT = Path(__file__).parent.parent / "outputs" / "human_eval"


def build_assignments(
    out_csv: str,
    evaluators: Optional[List[str]] = None,
    seed: int = SEED,
) -> None:
    """Write a randomised per-evaluator stimulus order to out_csv."""
    rng = random.Random(seed)
    evaluators = evaluators or DEFAULT_EVALUATORS
    rows = [["evaluator", "position", "stimulus_id", "video", "c_state"]]
    for ev in evaluators:
        order = list(STIMULI)
        rng.shuffle(order)
        for pos, (video, c) in enumerate(order, start=1):
            rows.append([ev, str(pos), f"{ev}-S{pos}", video, c])
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerows(rows)


def load_responses(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"evaluator", "stimulus_id", "video", "c_state",
                "q1_accuracy", "q2_usefulness"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"responses CSV missing columns: {missing}")
    if "email" in df.columns or "respondent_email" in df.columns:
        raise ValueError(
            "responses CSV contains a PII column (email). Strip it before "
            "loading; outputs/human_eval/ must not contain real names or addresses."
        )
    for col in ("q1_accuracy", "q2_usefulness"):
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)
        bad = df[(df[col] < 1) | (df[col] > 5)]
        if len(bad):
            raise ValueError(
                f"Out-of-range Likert in column {col}: {bad.to_dict('records')}"
            )
    if not df["c_state"].isin({"behavior", "counts"}).all():
        raise ValueError("c_state must be one of {'behavior', 'counts'}")
    return df


def aggregate(responses_df: pd.DataFrame) -> Dict[str, object]:
    cells = []
    for video, c_state in STIMULI:
        for q in QUESTIONS:
            sub = responses_df[
                (responses_df["video"] == video) &
                (responses_df["c_state"] == c_state)
            ][q]
            arr = sub.to_numpy(dtype=float)
            cells.append({
                "video": video, "c_state": c_state, "question": q,
                "n": int(arr.size),
                "mean": float(arr.mean()) if arr.size else None,
                "std": float(arr.std(ddof=1)) if arr.size > 1 else None,
                "values": [int(x) for x in arr],
            })

    signs: Dict[str, Dict[str, Dict[str, object]]] = {}
    for q in QUESTIONS:
        signs[q] = {}
        for video in ["MOT17-04-FRCNN", "MOT17-09-FRCNN", "demo_clip"]:
            on  = next(c for c in cells if c["video"] == video
                       and c["c_state"] == "behavior" and c["question"] == q)
            off = next(c for c in cells if c["video"] == video
                       and c["c_state"] == "counts"   and c["question"] == q)
            if on["mean"] is None or off["mean"] is None:
                signs[q][video] = {"c_on_minus_c_off_mean": None, "sign": "?"}
                continue
            d = on["mean"] - off["mean"]
            signs[q][video] = {
                "c_on_minus_c_off_mean": d,
                "sign": "+" if d > 0 else ("-" if d < 0 else "0"),
            }

    return {
        "per_cell": cells,
        "paired_signs": signs,
        "n_evaluators": int(responses_df["evaluator"].nunique()),
    }


def render_table(agg: Dict[str, object]) -> str:
    lines = [
        "| video | c_state | question | n | mean | std |",
        "|---|---|---|---:|---:|---:|",
    ]
    for c in agg["per_cell"]:
        m = "" if c["mean"] is None else f"{c['mean']:.2f}"
        s = "" if c["std"]  is None else f"{c['std']:.2f}"
        lines.append(
            f"| {c['video']} | {c['c_state']} | {c['question']} | "
            f"{c['n']} | {m} | {s} |"
        )
    lines += [
        "",
        "**Paired comparison (C-ON minus C-OFF, mean of evaluator means):**",
        "",
        "| question | video | delta (mean) | sign |",
        "|---|---|---:|:-:|",
    ]
    for q, by_vid in agg["paired_signs"].items():
        for vid, d in by_vid.items():
            v = ("" if d["c_on_minus_c_off_mean"] is None
                 else f"{d['c_on_minus_c_off_mean']:+.2f}")
            lines.append(f"| {q} | {vid} | {v} | {d['sign']} |")
    return "\n".join(lines) + "\n"



# CLI


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[human_eval] %(message)s")

    p = argparse.ArgumentParser(description="Phase 5 human-eval CLI.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("assignments", help="Generate evaluator assignment CSV.")
    s1.add_argument("--out-csv",
                    default=str(OUT_ROOT / "eval_assignments.csv"))

    s2 = sub.add_parser("aggregate", help="Aggregate evaluator responses.")
    s2.add_argument("--in-csv",
                    default=str(OUT_ROOT / "responses_raw.csv"))
    s2.add_argument("--out-json",
                    default=str(OUT_ROOT / "phase5_human_eval.json"))
    s2.add_argument("--out-md",
                    default=str(OUT_ROOT / "phase5_human_eval_table.md"))

    args = p.parse_args()

    if args.cmd == "assignments":
        build_assignments(args.out_csv)
        logger.info("assignments -> %s", args.out_csv)
        return 0

    if args.cmd == "aggregate":
        df = load_responses(args.in_csv)
        agg = aggregate(df)
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(agg, f, indent=2)
        Path(args.out_md).write_text(render_table(agg), encoding="utf-8")
        logger.info("aggregate -> %s", args.out_json)
        logger.info("aggregate -> %s", args.out_md)
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
