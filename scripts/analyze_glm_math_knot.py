"""
Analyze GLM math knot labels
============================
Summarize math GLM knot labels and quantify how knot prevalence and severity
relate to correctness.

Usage:
  python scripts/analyze_glm_math_knot.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parent.parent
JSONL_PATH = REPO_ROOT / "results" / "glm_math_knot_raw_v3" / "all_runs.jsonl"
OUT_LABELS = REPO_ROOT / "results" / "tables" / "glm_math_knot_labels_v3.csv"
OUT_SUMMARY = REPO_ROOT / "results" / "tables" / "glm_math_knot_summary_v3.csv"
OUT_SYMPTOMS = REPO_ROOT / "results" / "tables" / "glm_math_knot_symptoms_v3.csv"
OUT_DATASET = REPO_ROOT / "results" / "tables" / "glm_math_knot_dataset_v3.csv"

SYMPTOMS = [
    "case_split_instability",
    "assumption_drift",
    "subgoal_frame_loss",
    "variable_binding_drift",
    "invariant_drift",
    "repair_without_recovery",
]


def _coerce_symptoms(value) -> list[str]:
    if isinstance(value, list):
        out = [str(item).strip() for item in value if str(item).strip()]
    elif isinstance(value, str):
        out = [value.strip()] if value.strip() else []
    else:
        out = []
    if not out:
        return ["none"]
    filtered = [item for item in out if item != "none"]
    return filtered or ["none"]


def load_labels(jsonl_path: Path) -> pd.DataFrame:
    rows = []
    with open(jsonl_path) as handle:
        for line in handle:
            record = json.loads(line)
            parsed = record.get("glm_parsed", {})
            if not parsed:
                continue
            symptoms = _coerce_symptoms(parsed.get("knot_symptoms", ["none"]))
            row = {
                "dataset": record["dataset"],
                "problem_id": record["problem_id"],
                "problem_key": record["problem_key"],
                "run_index": int(record["run_index"]),
                "is_correct": int(record["is_correct"]),
                "think_chars": int(record.get("think_chars", 0)),
                "knot_present": str(parsed.get("knot_present", "")).strip().lower(),
                "knot_present_bin": 1 if str(parsed.get("knot_present", "")).strip().lower() == "yes" else 0,
                "knot_severity": int(parsed.get("knot_severity", 0) or 0),
                "primary_trigger": str(parsed.get("primary_trigger", "")),
                "trace_strategy": str(parsed.get("trace_strategy", "")),
                "reversal_count": int(parsed.get("reversal_count", 0) or 0),
                "state_consistency": str(parsed.get("state_consistency", "")),
                "knot_quote": str(parsed.get("knot_quote", "")),
                "open_diagnosis": str(parsed.get("open_diagnosis", "")),
                "knot_symptoms": "|".join(symptoms),
                "knot_count": 0 if symptoms == ["none"] else len(symptoms),
            }
            for symptom in SYMPTOMS:
                row[f"sym_{symptom}"] = int(symptom in symptoms)
            rows.append(row)
    return pd.DataFrame(rows)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[df["is_correct"].isin([0, 1])].copy()
    if valid["knot_severity"].nunique() <= 1 or valid["is_correct"].nunique() <= 1:
        rho_sev, p_sev = np.nan, np.nan
    else:
        rho_sev, p_sev = spearmanr(valid["knot_severity"], valid["is_correct"])
    if valid["reversal_count"].nunique() <= 1 or valid["is_correct"].nunique() <= 1:
        rho_rev, p_rev = np.nan, np.nan
    else:
        rho_rev, p_rev = spearmanr(valid["reversal_count"], valid["is_correct"])
    grouped = valid.groupby("is_correct").agg(
        n=("run_index", "count"),
        knot_rate=("knot_present_bin", "mean"),
        mean_severity=("knot_severity", "mean"),
        mean_reversal_count=("reversal_count", "mean"),
        mean_think_chars=("think_chars", "mean"),
    )
    rows = []
    for label in [1, 0]:
        if label not in grouped.index:
            continue
        g = grouped.loc[label]
        rows.append({
            "group": "correct" if label == 1 else "incorrect",
            "n": int(g["n"]),
            "knot_rate": float(g["knot_rate"]),
            "mean_severity": float(g["mean_severity"]),
            "mean_reversal_count": float(g["mean_reversal_count"]),
            "mean_think_chars": float(g["mean_think_chars"]),
            "rho_knot_severity_correct": float(rho_sev),
            "p_knot_severity_correct": float(p_sev),
            "rho_reversal_count_correct": float(rho_rev),
            "p_reversal_count_correct": float(p_rev),
        })
    return pd.DataFrame(rows)


def build_symptom_table(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[df["is_correct"].isin([0, 1])].copy()
    correct = valid[valid["is_correct"] == 1]
    incorrect = valid[valid["is_correct"] == 0]
    rows = []
    for symptom in SYMPTOMS:
        col = f"sym_{symptom}"
        rate_correct = float(correct[col].mean()) if len(correct) else float("nan")
        rate_incorrect = float(incorrect[col].mean()) if len(incorrect) else float("nan")
        rows.append({
            "symptom": symptom,
            "rate_correct": rate_correct,
            "rate_incorrect": rate_incorrect,
            "diff_incorrect_minus_correct": rate_incorrect - rate_correct,
        })
    return pd.DataFrame(rows).sort_values("diff_incorrect_minus_correct", ascending=False)


def build_dataset_table(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[df["is_correct"].isin([0, 1])].copy()
    return (
        valid.groupby("dataset")
        .agg(
            n=("run_index", "count"),
            accuracy=("is_correct", "mean"),
            knot_rate=("knot_present_bin", "mean"),
            mean_severity=("knot_severity", "mean"),
            mean_reversal_count=("reversal_count", "mean"),
        )
        .reset_index()
        .sort_values(["knot_rate", "mean_severity"], ascending=False)
    )


def main():
    parser = argparse.ArgumentParser(description="Analyze GLM math knot labels")
    parser.add_argument("--jsonl-path", type=str, default=str(JSONL_PATH))
    parser.add_argument("--out-labels", type=str, default=str(OUT_LABELS))
    parser.add_argument("--out-summary", type=str, default=str(OUT_SUMMARY))
    parser.add_argument("--out-symptoms", type=str, default=str(OUT_SYMPTOMS))
    parser.add_argument("--out-dataset", type=str, default=str(OUT_DATASET))
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl_path)
    if not jsonl_path.exists():
        print(f"ERROR: {jsonl_path} not found.")
        print("Run: GLM_API_KEY=<key> python scripts/run_glm_math_knot_labeling.py")
        return

    df = load_labels(jsonl_path)
    print(f"Loaded {len(df)} labeled runs from {jsonl_path}")
    if df.empty:
        print("No parsed labels found.")
        return

    Path(args.out_labels).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_labels, index=False)

    summary = build_summary(df)
    symptoms = build_symptom_table(df)
    dataset_table = build_dataset_table(df)
    summary.to_csv(args.out_summary, index=False)
    symptoms.to_csv(args.out_symptoms, index=False)
    dataset_table.to_csv(args.out_dataset, index=False)

    print(f"Saved labels   -> {args.out_labels}")
    print(f"Saved summary  -> {args.out_summary}")
    print(f"Saved symptoms -> {args.out_symptoms}")
    print(f"Saved dataset  -> {args.out_dataset}")

    print("\n=== Summary ===")
    print(summary.to_string(index=False, float_format=lambda value: f'{value:.4f}'))
    print("\n=== Symptoms (incorrect - correct) ===")
    print(symptoms.to_string(index=False, float_format=lambda value: f'{value:.4f}'))
    print("\n=== Trigger x correctness ===")
    print(pd.crosstab(df["primary_trigger"], df["is_correct"]).to_string())
    print("\n=== State x correctness ===")
    print(pd.crosstab(df["state_consistency"], df["is_correct"]).to_string())


if __name__ == "__main__":
    main()
