"""
Analyze GLM science (GPQA) knot labels v2
==========================================
Summarize GLM science knot labels from the v2 replacement protocol
and compare against v1.

Usage:
  python scripts/analyze_glm_science_knot_v2.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parent.parent
V1_JSONL = REPO_ROOT / "results" / "glm_science_knot_raw_v1" / "all_runs.jsonl"
V2_JSONL = REPO_ROOT / "results" / "glm_science_knot_raw_v2" / "all_runs.jsonl"
OUT_LABELS = REPO_ROOT / "results" / "tables" / "glm_science_knot_labels_v2.csv"
OUT_SUMMARY = REPO_ROOT / "results" / "tables" / "glm_science_knot_summary_v2.csv"
OUT_COMPARISON = REPO_ROOT / "results" / "tables" / "glm_science_knot_protocol_comparison.csv"

V2_SYMPTOMS = [
    "explicit_claim_reversal",
    "framework_collision",
    "formula_scope_violation",
    "irreparable_regression",
]


def load_labels(jsonl_path: Path, symptom_keys: list[str]) -> pd.DataFrame:
    rows = []
    with open(jsonl_path) as handle:
        for line in handle:
            record = json.loads(line)
            parsed = record.get("glm_parsed", {})
            if not parsed:
                continue
            symptoms_raw = parsed.get("knot_symptoms", [])
            if isinstance(symptoms_raw, list):
                symptoms = [str(s).strip() for s in symptoms_raw if str(s).strip() and str(s).strip() != "none"]
            else:
                symptoms = []
            row = {
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
                "parse_error": record.get("parse_error"),
            }
            for sk in symptom_keys:
                row[f"sym_{sk}"] = int(sk in symptoms)
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, label: str) -> dict:
    total = len(df)
    n_pos = int(df["knot_present_bin"].sum())
    prevalence = n_pos / total if total else float("nan")
    pos_correct = df[df["is_correct"] == 1]["knot_present_bin"].mean() if (df["is_correct"] == 1).any() else float("nan")
    pos_incorrect = df[df["is_correct"] == 0]["knot_present_bin"].mean() if (df["is_correct"] == 0).any() else float("nan")

    if n_pos >= 2:
        rho, pval = spearmanr(df["knot_present_bin"], df["is_correct"])
    else:
        rho, pval = float("nan"), float("nan")

    return {
        "protocol": label,
        "total_runs": total,
        "n_positive": n_pos,
        "prevalence": round(prevalence, 4),
        "pos_in_correct": round(pos_correct, 4),
        "pos_in_incorrect": round(pos_incorrect, 4),
        "spearman_rho": round(rho, 4),
        "spearman_pval": round(pval, 4) if not np.isnan(pval) else float("nan"),
        "mean_severity": round(df["knot_severity"].mean(), 4),
        "degenerate": prevalence > 0.90 or prevalence < 0.03,
    }


def print_report(df_v2: pd.DataFrame, comp: pd.DataFrame):
    print("\n" + "=" * 65)
    print("SCIENCE KNOT ANNOTATION — PROTOCOL COMPARISON")
    print("=" * 65)
    print(comp.to_string(index=False))
    print()

    v1_row = comp[comp["protocol"] == "v1_exploratory"]
    v2_row = comp[comp["protocol"] == "v2_replacement"]

    print("INTERPRETATION:")
    if not v1_row.empty:
        prev = v1_row["prevalence"].values[0]
        if prev > 0.90:
            print(f"  v1 (exploratory): {prev:.0%} positive → DEGENERATE COLLAPSE")
            print("    Analogous to coding original protocol (κ = 0.0)")

    if not v2_row.empty:
        prev = v2_row["prevalence"].values[0]
        rho = v2_row["spearman_rho"].values[0]
        pval = v2_row["spearman_pval"].values[0]
        if prev < 0.10:
            print(f"  v2 (replacement): {prev:.0%} positive → TOO SPARSE for κ estimation")
        else:
            print(f"  v2 (replacement): {prev:.0%} positive")
        print(f"    ρ(knot_bin vs correct) = {rho:.3f}, p = {pval:.4f}")
        if not np.isnan(rho) and rho > 0:
            print("    NOTE: knots more prevalent in CORRECT runs (unexpected direction)")

    print()
    print("CONCLUSION for paper:")
    print("  Both science annotation protocols fail in different ways.")
    print("  v1: degenerate (>90% positive) — concept too broad for science domain")
    print("  v2: sparse (<10% positive) or biased direction — concept too narrow")
    print("  Science annotation validity: — (requires different methodological approach)")
    print()
    print("  This mirrors the coding annotation challenge but with opposite failure mode:")
    print("  Math:    κ = 0.603 (valid, ~53% positive, negatively correlated)")
    print("  Science: v1 ~100% → degenerate; v2 ~9% → sparse/wrong direction")
    print("  Coding:  κ = 0.0 (degenerate, original); κ = 0.329 (replacement, ~98%)")
    print("=" * 65)

    print("\nv2 positive traces:")
    pos = df_v2[df_v2["knot_present_bin"] == 1]
    for _, row in pos.iterrows():
        print(f"  [{row['problem_key']} run{row['run_index']}] correct={row['is_correct']} "
              f"sev={row['knot_severity']} trigger={row['primary_trigger']}")
        if row["knot_quote"]:
            print(f"    quote: {row['knot_quote'][:90]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-jsonl", default=str(V1_JSONL))
    parser.add_argument("--v2-jsonl", default=str(V2_JSONL))
    args = parser.parse_args()

    Path(OUT_LABELS).parent.mkdir(parents=True, exist_ok=True)

    rows_comp = []

    if Path(args.v1_jsonl).exists():
        df_v1 = load_labels(Path(args.v1_jsonl), [])
        rows_comp.append(summarize(df_v1, "v1_exploratory"))
    else:
        print(f"v1 JSONL not found at {args.v1_jsonl}")
        df_v1 = pd.DataFrame()

    if Path(args.v2_jsonl).exists():
        df_v2 = load_labels(Path(args.v2_jsonl), V2_SYMPTOMS)
        rows_comp.append(summarize(df_v2, "v2_replacement"))
        df_v2.to_csv(OUT_LABELS, index=False)
        print(f"Saved v2 labels: {OUT_LABELS}")

        # Save v2 summary
        summary_rows = []
        for metric, val in rows_comp[-1].items():
            summary_rows.append({"metric": metric, "value": val})
        pd.DataFrame(summary_rows).to_csv(OUT_SUMMARY, index=False)
        print(f"Saved v2 summary: {OUT_SUMMARY}")
    else:
        print(f"v2 JSONL not found at {args.v2_jsonl}")
        df_v2 = pd.DataFrame()

    if rows_comp:
        comp_df = pd.DataFrame(rows_comp)
        comp_df.to_csv(OUT_COMPARISON, index=False)
        print(f"Saved comparison: {OUT_COMPARISON}")
        print_report(df_v2 if not df_v2.empty else pd.DataFrame(), comp_df)


if __name__ == "__main__":
    main()
