"""
Analyze GLM math knot spans
===========================
Summarize span-extraction outputs from `run_glm_math_knot_spans.py`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parent.parent
JSONL_PATH = REPO_ROOT / "results" / "glm_math_knot_span_raw_v1" / "all_runs.jsonl"
OUT_LABELS = REPO_ROOT / "results" / "tables" / "glm_math_knot_span_labels_v1.csv"
OUT_SUMMARY = REPO_ROOT / "results" / "tables" / "glm_math_knot_span_summary_v1.csv"

SYMPTOMS = [
    "case_split_instability",
    "assumption_drift",
    "subgoal_frame_loss",
    "variable_binding_drift",
    "invariant_drift",
    "repair_without_recovery",
]


def load_rows(path: Path) -> pd.DataFrame:
    rows = []
    with open(path) as handle:
        for line in handle:
            record = json.loads(line)
            parsed = record.get("glm_parsed", {}) or {}
            spans = parsed.get("explicit_knot_spans", [])
            if not isinstance(spans, list):
                spans = []
            symptoms = [span.get("symptom", "") for span in spans if isinstance(span, dict)]
            row = {
                "dataset": record["dataset"],
                "problem_id": record["problem_id"],
                "run_index": int(record["run_index"]),
                "is_correct": int(record["is_correct"]),
                "n_spans": len(spans),
                "has_knot_span": int(len(spans) > 0),
                "trace_strategy": str(parsed.get("trace_strategy", "")),
                "overall_state": str(parsed.get("overall_state", "")),
            }
            for symptom in SYMPTOMS:
                row[f"sym_{symptom}"] = int(symptom in symptoms)
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Analyze math knot span labels")
    parser.add_argument("--jsonl-path", type=str, default=str(JSONL_PATH))
    parser.add_argument("--out-labels", type=str, default=str(OUT_LABELS))
    parser.add_argument("--out-summary", type=str, default=str(OUT_SUMMARY))
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl_path)
    df = load_rows(jsonl_path)
    df.to_csv(args.out_labels, index=False)

    if df["has_knot_span"].nunique() <= 1 or df["is_correct"].nunique() <= 1:
        rho, p_val = np.nan, np.nan
    else:
        rho, p_val = spearmanr(df["has_knot_span"], df["is_correct"])

    summary = (
        df.groupby("is_correct")
        .agg(
            n=("run_index", "count"),
            knot_span_rate=("has_knot_span", "mean"),
            mean_n_spans=("n_spans", "mean"),
        )
        .reset_index()
    )
    summary["rho_has_knot_span_correct"] = rho
    summary["p_has_knot_span_correct"] = p_val
    summary.to_csv(args.out_summary, index=False)

    print(df.to_string(index=False))
    print("\n=== Summary ===")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
