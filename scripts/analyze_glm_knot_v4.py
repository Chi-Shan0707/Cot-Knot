"""
Analyze GLM knot labels v4
==========================
Summarize one domain's v4 knot labels into stable CSV tables.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from knot_glm_common import REPO_ROOT
from knot_v4_configs import DOMAIN_CONFIGS


def _coerce_symptoms(value, allowed: tuple[str, ...]) -> list[str]:
    if isinstance(value, list):
        out = [str(item).strip() for item in value if str(item).strip()]
    elif isinstance(value, str):
        out = [item.strip() for item in value.split("|") if item.strip()]
    else:
        out = []
    filtered = [item for item in out if item in allowed]
    return filtered or ["none"]


def _safe_int(value):
    try:
        return int(value)
    except Exception:
        return np.nan


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


SCIENCE_HEDGE_PREFIX = re.compile(r"^\s*(perhaps|maybe|i think|probably|likely|could)\b", re.I)
SCIENCE_STRONG_RE = re.compile(
    r"\b(doesn['’]?t|cannot|can['’]?t|inconsistent|contradict|impossible|same proton|same signal|still not|not .* but|expected .* but)\b",
    re.I,
)
CODING_RESTATEMENT_PREFIX = re.compile(
    r"^\s*(the problem asks|the operation is defined|the statement allows|the example explanation says|the minimal steps)\b",
    re.I,
)
CODING_STATE_RE = re.compile(r"\b(state|index|pointer|range|query|branch|loop|variable|window)\b", re.I)
CODING_STRONG_RE = re.compile(
    r"\b(not matching|wrong|inconsistent|defined by|same state|but then|however|cannot|can['’]?t|doesn['’]?t)\b",
    re.I,
)
MATH_SETUP_PREFIX = re.compile(r"^\s*(let me set|let me denote|suppose)\b", re.I)
MATH_STRONG_RE = re.compile(r"\b(contradict|cannot|can['’]?t|but then|however|still not)\b", re.I)


def apply_protocol_filter(domain: str, raw_present: str, quote: str, reversal_count, state_consistency: str) -> tuple[bool, str]:
    if raw_present != "yes":
        return False, "raw_no"

    quote = (quote or "").strip()
    quote_lower = quote.lower()
    reversal_count = int(reversal_count) if not pd.isna(reversal_count) else 0
    state_consistency = str(state_consistency or "")

    if domain == "math":
        if MATH_SETUP_PREFIX.match(quote_lower) and reversal_count < 2 and not MATH_STRONG_RE.search(quote_lower):
            return False, "setup_search"
        return True, "accepted_math"

    if domain == "science":
        if SCIENCE_HEDGE_PREFIX.match(quote_lower) and reversal_count < 3:
            return False, "hedged_hypothesis"
        if state_consistency in {"lost_state", "self_contradictory"} and (SCIENCE_STRONG_RE.search(quote_lower) or reversal_count >= 3):
            return True, "science_strong_lost_state"
        if reversal_count >= 3 and SCIENCE_STRONG_RE.search(quote_lower):
            return True, "science_repair_plus_contradiction"
        return False, "science_no_hard_evidence"

    if domain == "coding":
        if CODING_RESTATEMENT_PREFIX.match(quote_lower):
            return False, "restatement_or_sample"
        if reversal_count >= 3 and CODING_STATE_RE.search(quote_lower) and CODING_STRONG_RE.search(quote_lower):
            return True, "coding_state_conflict"
        if state_consistency in {"lost_state", "self_contradictory"} and CODING_STATE_RE.search(quote_lower):
            return True, "coding_lost_state"
        return False, "coding_no_hard_evidence"

    return False, "unknown_domain"


def load_labels(jsonl_path: Path, domain: str, symptoms: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    with open(jsonl_path) as handle:
        for line in handle:
            record = json.loads(line)
            parsed = record.get("glm_parsed") or {}
            parse_ok = int(bool(parsed) and not record.get("parse_error"))
            symptom_list = _coerce_symptoms(parsed.get("knot_symptoms", []), symptoms) if parse_ok else ["none"]
            raw_knot_present = str(parsed.get("knot_present", "")).strip().lower() if parse_ok else ""
            raw_knot_severity = _safe_int(parsed.get("knot_severity")) if parse_ok else np.nan
            raw_primary_trigger = str(parsed.get("primary_trigger", "")) if parse_ok else ""
            raw_trace_strategy = str(parsed.get("trace_strategy", "")) if parse_ok else ""
            raw_reversal_count = _safe_int(parsed.get("reversal_count")) if parse_ok else np.nan
            raw_state_consistency = str(parsed.get("state_consistency", "")) if parse_ok else ""
            raw_recovers_later = str(parsed.get("recovers_later", "")) if parse_ok else ""
            raw_annotator_confidence = str(parsed.get("annotator_confidence", "")) if parse_ok else ""
            raw_knot_quote = str(parsed.get("knot_quote", "")) if parse_ok else ""
            raw_open_diagnosis = str(parsed.get("open_diagnosis", "")) if parse_ok else ""
            filtered_positive, filter_reason = apply_protocol_filter(
                domain,
                raw_knot_present,
                raw_knot_quote,
                raw_reversal_count,
                raw_state_consistency,
            )
            effective_symptoms = symptom_list if filtered_positive else ["none"]
            row = {
                "domain": domain,
                "dataset": record.get("dataset", ""),
                "problem_id": record.get("problem_id", ""),
                "problem_key": record.get("problem_key", ""),
                "run_index": _safe_int(record.get("run_index")),
                "is_correct": _safe_int(record.get("is_correct")),
                "parse_ok": parse_ok,
                "think_chars": _safe_int(record.get("think_chars")),
                "think_total_chars": _safe_int(record.get("think_total_chars")),
                "raw_knot_present": raw_knot_present,
                "raw_knot_severity": raw_knot_severity,
                "raw_primary_trigger": raw_primary_trigger,
                "raw_trace_strategy": raw_trace_strategy,
                "raw_reversal_count": raw_reversal_count,
                "raw_state_consistency": raw_state_consistency,
                "raw_recovers_later": raw_recovers_later,
                "raw_annotator_confidence": raw_annotator_confidence,
                "raw_knot_quote": raw_knot_quote,
                "raw_open_diagnosis": raw_open_diagnosis,
                "raw_knot_symptoms": "|".join(symptom_list),
                "protocol_filter": filter_reason,
                "knot_present": "yes" if filtered_positive else "no",
                "knot_present_bin": int(filtered_positive),
                "knot_severity": raw_knot_severity if filtered_positive else 0,
                "primary_trigger": raw_primary_trigger if filtered_positive else "none",
                "trace_strategy": raw_trace_strategy,
                "reversal_count": raw_reversal_count,
                "state_consistency": raw_state_consistency,
                "recovers_later": raw_recovers_later,
                "annotator_confidence": raw_annotator_confidence,
                "knot_quote": raw_knot_quote,
                "open_diagnosis": raw_open_diagnosis,
                "knot_symptoms": "|".join(effective_symptoms),
                "knot_count": 0 if effective_symptoms == ["none"] else len(effective_symptoms),
            }
            for symptom in symptoms:
                row[f"sym_{symptom}"] = int(symptom in effective_symptoms)
            rows.append(row)
    return pd.DataFrame(rows)


def _safe_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    valid = pd.concat([x, y], axis=1).dropna()
    if len(valid) == 0:
        return np.nan, np.nan
    if valid.iloc[:, 0].nunique() <= 1 or valid.iloc[:, 1].nunique() <= 1:
        return np.nan, np.nan
    rho, pval = spearmanr(valid.iloc[:, 0], valid.iloc[:, 1])
    return float(rho), float(pval)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[(df["parse_ok"] == 1) & (df["is_correct"].isin([0, 1]))].copy()
    correct = valid[valid["is_correct"] == 1]
    incorrect = valid[valid["is_correct"] == 0]
    rho_knot, p_knot = _safe_spearman(valid["knot_present_bin"], valid["is_correct"])
    rho_sev, p_sev = _safe_spearman(valid["knot_severity"], valid["is_correct"])
    summary = {
        "domain": df["domain"].iloc[0] if len(df) else "",
        "total_runs": int(len(df)),
        "n_unique_problems": int(df["problem_key"].nunique()),
        "parse_ok_rate": float(df["parse_ok"].mean()) if len(df) else np.nan,
        "n_correct": int((valid["is_correct"] == 1).sum()),
        "n_incorrect": int((valid["is_correct"] == 0).sum()),
        "pct_correct": float(valid["is_correct"].mean()) if len(valid) else np.nan,
        "n_knot_present": int(valid["knot_present_bin"].sum()) if len(valid) else 0,
        "knot_prevalence": float(valid["knot_present_bin"].mean()) if len(valid) else np.nan,
        "knot_rate_correct": float(correct["knot_present_bin"].mean()) if len(correct) else np.nan,
        "knot_rate_incorrect": float(incorrect["knot_present_bin"].mean()) if len(incorrect) else np.nan,
        "mean_knot_severity": float(valid["knot_severity"].mean()) if len(valid) else np.nan,
        "mean_reversal_count": float(valid["reversal_count"].mean()) if len(valid) else np.nan,
        "spearman_rho_knot_vs_correct": rho_knot,
        "spearman_pval_knot_vs_correct": p_knot,
        "spearman_rho_severity_vs_correct": rho_sev,
        "spearman_pval_severity_vs_correct": p_sev,
        "degenerate_flag": int(valid["knot_present_bin"].nunique() <= 1) if len(valid) else 1,
    }
    return pd.DataFrame([summary])


def build_symptom_table(df: pd.DataFrame, symptoms: tuple[str, ...]) -> pd.DataFrame:
    valid = df[(df["parse_ok"] == 1) & (df["is_correct"].isin([0, 1]))].copy()
    correct = valid[valid["is_correct"] == 1]
    incorrect = valid[valid["is_correct"] == 0]
    rows = []
    for symptom in symptoms:
        col = f"sym_{symptom}"
        rho, pval = _safe_spearman(valid[col], valid["is_correct"])
        rows.append(
            {
                "domain": df["domain"].iloc[0] if len(df) else "",
                "symptom": symptom,
                "n_present": int(valid[col].sum()) if len(valid) else 0,
                "prevalence": float(valid[col].mean()) if len(valid) else np.nan,
                "rate_correct": float(correct[col].mean()) if len(correct) else np.nan,
                "rate_incorrect": float(incorrect[col].mean()) if len(incorrect) else np.nan,
                "diff_incorrect_minus_correct": (
                    float(incorrect[col].mean()) - float(correct[col].mean())
                    if len(correct) and len(incorrect)
                    else np.nan
                ),
                "spearman_rho_vs_correct": rho,
                "spearman_pval_vs_correct": pval,
            }
        )
    return pd.DataFrame(rows).sort_values(["prevalence", "n_present"], ascending=False)


def default_paths(domain: str) -> tuple[Path, Path, Path, Path]:
    stem = f"glm_{domain}_knot"
    jsonl_path = DOMAIN_CONFIGS[domain].default_out_dir / "all_runs.jsonl"
    out_dir = REPO_ROOT / "results" / "tables"
    return (
        jsonl_path,
        out_dir / f"{stem}_labels_v4.csv",
        out_dir / f"{stem}_summary_v4.csv",
        out_dir / f"{stem}_symptoms_v4.csv",
    )


def main():
    parser = argparse.ArgumentParser(description="Analyze GLM knot labels v4")
    parser.add_argument("--domain", choices=sorted(DOMAIN_CONFIGS), required=True)
    parser.add_argument("--jsonl-path", type=str, default="")
    parser.add_argument("--out-labels", type=str, default="")
    parser.add_argument("--out-summary", type=str, default="")
    parser.add_argument("--out-symptoms", type=str, default="")
    args = parser.parse_args()

    cfg = DOMAIN_CONFIGS[args.domain]
    default_jsonl, default_labels, default_summary, default_symptoms = default_paths(args.domain)
    jsonl_path = Path(args.jsonl_path) if args.jsonl_path else default_jsonl
    out_labels = Path(args.out_labels) if args.out_labels else default_labels
    out_summary = Path(args.out_summary) if args.out_summary else default_summary
    out_symptoms = Path(args.out_symptoms) if args.out_symptoms else default_symptoms

    if not jsonl_path.exists():
        print(f"ERROR: {jsonl_path} not found.")
        return

    df = load_labels(jsonl_path, args.domain, cfg.symptoms)
    print(f"Loaded {len(df)} records from {jsonl_path}")
    if df.empty:
        print("No records found.")
        return

    out_labels.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_labels, index=False)

    summary = build_summary(df)
    symptoms = build_symptom_table(df, cfg.symptoms)
    summary.to_csv(out_summary, index=False)
    symptoms.to_csv(out_symptoms, index=False)

    print(f"Saved labels   -> {out_labels}")
    print(f"Saved summary  -> {out_summary}")
    print(f"Saved symptoms -> {out_symptoms}")
    print("\n=== Summary ===")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\n=== Top symptoms ===")
    print(symptoms.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
