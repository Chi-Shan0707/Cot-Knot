"""
Analyze GLM science (GPQA) knot labels
=======================================
Summarize GLM science knot labels and quantify how knot prevalence and
severity relate to correctness.

Analogous to analyze_glm_math_knot.py but for science symptoms.

Usage:
  python scripts/analyze_glm_science_knot.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu


REPO_ROOT = Path(__file__).resolve().parent.parent
JSONL_PATH = REPO_ROOT / "results" / "glm_science_knot_raw_v1" / "all_runs.jsonl"
OUT_LABELS = REPO_ROOT / "results" / "tables" / "glm_science_knot_labels_v1.csv"
OUT_SUMMARY = REPO_ROOT / "results" / "tables" / "glm_science_knot_summary_v1.csv"
OUT_SYMPTOMS = REPO_ROOT / "results" / "tables" / "glm_science_knot_symptoms_v1.csv"

SYMPTOMS = [
    "concept_conflation",
    "formula_scope_violation",
    "variable_identity_drift",
    "assumption_contradiction",
    "causal_chain_break",
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


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    n_correct = df["is_correct"].sum()
    n_knot = df["knot_present_bin"].sum()
    knot_rate = n_knot / total if total else float("nan")

    # Knot prevalence by correctness
    knot_correct = df[df["is_correct"] == 1]["knot_present_bin"].mean()
    knot_incorrect = df[df["is_correct"] == 0]["knot_present_bin"].mean()

    # Correlation: knot_present_bin vs is_correct
    if len(df) >= 5:
        rho, pval = spearmanr(df["knot_present_bin"], df["is_correct"])
    else:
        rho, pval = float("nan"), float("nan")

    # Severity vs correctness
    if len(df) >= 5:
        rho_sev, pval_sev = spearmanr(df["knot_severity"], df["is_correct"])
    else:
        rho_sev, pval_sev = float("nan"), float("nan")

    rows = [
        {"metric": "total_runs", "value": total},
        {"metric": "n_correct", "value": n_correct},
        {"metric": "n_incorrect", "value": total - n_correct},
        {"metric": "pct_correct", "value": round(n_correct / total, 4) if total else float("nan")},
        {"metric": "n_knot_present", "value": n_knot},
        {"metric": "knot_prevalence", "value": round(knot_rate, 4)},
        {"metric": "knot_rate_correct", "value": round(knot_correct, 4)},
        {"metric": "knot_rate_incorrect", "value": round(knot_incorrect, 4)},
        {"metric": "spearman_rho_knot_vs_correct", "value": round(rho, 4)},
        {"metric": "spearman_pval_knot_vs_correct", "value": round(pval, 6) if not np.isnan(pval) else float("nan")},
        {"metric": "spearman_rho_severity_vs_correct", "value": round(rho_sev, 4)},
        {"metric": "spearman_pval_severity_vs_correct", "value": round(pval_sev, 6) if not np.isnan(pval_sev) else float("nan")},
        {"metric": "mean_knot_severity", "value": round(df["knot_severity"].mean(), 4)},
        {"metric": "mean_reversal_count", "value": round(df["reversal_count"].mean(), 4)},
        {"metric": "parse_ok_rate", "value": round(len(df) / max(len(df), 1), 4)},
    ]
    return pd.DataFrame(rows)


def compute_symptoms(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for symptom in SYMPTOMS:
        col = f"sym_{symptom}"
        if col not in df.columns:
            continue
        n_present = int(df[col].sum())
        rate = n_present / len(df) if len(df) else float("nan")

        # Symptom vs correctness correlation
        if n_present >= 3:
            rho, pval = spearmanr(df[col], df["is_correct"])
        else:
            rho, pval = float("nan"), float("nan")

        rows.append({
            "symptom": symptom,
            "n_present": n_present,
            "prevalence": round(rate, 4),
            "spearman_rho_vs_correct": round(rho, 4),
            "spearman_pval_vs_correct": round(pval, 6) if not np.isnan(pval) else float("nan"),
        })
    return pd.DataFrame(rows)


def print_report(df: pd.DataFrame, summary: pd.DataFrame, symptoms: pd.DataFrame):
    print("\n" + "=" * 60)
    print("GLM SCIENCE KNOT ANNOTATION SUMMARY")
    print("=" * 60)
    print(f"Total runs:        {len(df)}")
    print(f"Problems:          {df['problem_key'].nunique()}")
    print(f"Correct runs:      {df['is_correct'].sum()} ({df['is_correct'].mean():.1%})")
    print()

    n_knot = df["knot_present_bin"].sum()
    print(f"Knot prevalence:   {n_knot}/{len(df)} = {n_knot/len(df):.1%}")
    knot_correct = df[df["is_correct"] == 1]["knot_present_bin"].mean()
    knot_incorrect = df[df["is_correct"] == 0]["knot_present_bin"].mean()
    print(f"  In correct runs: {knot_correct:.1%}")
    print(f"  In incorrect:    {knot_incorrect:.1%}")

    rho_row = summary[summary["metric"] == "spearman_rho_knot_vs_correct"]
    pval_row = summary[summary["metric"] == "spearman_pval_knot_vs_correct"]
    if not rho_row.empty:
        print(f"\nKnot × correctness: ρ = {rho_row['value'].values[0]:.4f}, p = {pval_row['value'].values[0]:.4f}")

    rho_sev_row = summary[summary["metric"] == "spearman_rho_severity_vs_correct"]
    pval_sev_row = summary[summary["metric"] == "spearman_pval_severity_vs_correct"]
    if not rho_sev_row.empty:
        print(f"Severity × correct: ρ = {rho_sev_row['value'].values[0]:.4f}, p = {pval_sev_row['value'].values[0]:.4f}")

    print("\nSymptom breakdown:")
    print(symptoms.to_string(index=False))

    print("\nTrace strategy distribution:")
    print(df["trace_strategy"].value_counts().to_string())

    print("\nState consistency distribution:")
    print(df["state_consistency"].value_counts().to_string())

    print("\nPrimary trigger distribution:")
    print(df["primary_trigger"].value_counts().to_string())

    print("\nKnot severity distribution:")
    print(df["knot_severity"].value_counts().sort_index().to_string())

    # Knot-positive subset geometry (analogous to paper's break-positive analysis)
    knot_pos = df[df["knot_present_bin"] == 1]
    if len(knot_pos) >= 5:
        n_kp_correct = knot_pos["is_correct"].sum()
        n_kp_incorrect = (knot_pos["is_correct"] == 0).sum()
        print(f"\nKnot-positive subset: n={len(knot_pos)} ({n_kp_correct} correct, {n_kp_incorrect} incorrect)")
        rho_kp, pval_kp = spearmanr(knot_pos["knot_severity"], knot_pos["is_correct"])
        print(f"  Severity × correct (knot-positive): ρ = {rho_kp:.4f}, p = {pval_kp:.4f}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Analyze GLM science knot labels")
    parser.add_argument("--jsonl", type=str, default=str(JSONL_PATH))
    parser.add_argument("--out-labels", type=str, default=str(OUT_LABELS))
    parser.add_argument("--out-summary", type=str, default=str(OUT_SUMMARY))
    parser.add_argument("--out-symptoms", type=str, default=str(OUT_SYMPTOMS))
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        print(f"ERROR: {jsonl_path} not found. Run run_glm_science_knot_labeling.py first.")
        return

    df = load_labels(jsonl_path)
    if df.empty:
        print("No valid parsed records found.")
        return

    summary = compute_summary(df)
    symptoms = compute_symptoms(df)

    # Ensure output directory exists
    Path(args.out_labels).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_labels, index=False)
    summary.to_csv(args.out_summary, index=False)
    symptoms.to_csv(args.out_symptoms, index=False)

    print_report(df, summary, symptoms)
    print(f"\nSaved:")
    print(f"  {args.out_labels}")
    print(f"  {args.out_summary}")
    print(f"  {args.out_symptoms}")


if __name__ == "__main__":
    main()
