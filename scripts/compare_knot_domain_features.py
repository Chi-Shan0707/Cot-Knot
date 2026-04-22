"""
Compare math-vs-coding knot runs in shared cache features
=========================================================

Build a unified run table for:
- Math: GLM extracted + GLM verified explicit knot spans (pilot64)
- Coding: existing GLM knot labels (partial/partial96/smoke/smoke3)

Then compare shared cache-based trajectory / activation features across domains.

Outputs:
  - results/tables/knot_cross_domain_run_table_pilot64.csv
  - results/tables/knot_cross_domain_feature_effects_pilot64.csv
  - results/tables/knot_cross_domain_temporal_profiles_pilot64.csv
  - results/tables/knot_cross_domain_label_patterns_pilot64.csv
"""

from __future__ import annotations

import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


REPO_ROOT = Path(__file__).resolve().parent.parent
NAD_ROOT = REPO_ROOT.parent / "NAD_Next"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(NAD_ROOT))

from nad.ops.earlystop_svd import FULL_FEATURE_NAMES  # noqa: E402


CACHE_PATH = NAD_ROOT / "results/cache/es_svd_ms_rr_r1/cache_all_547b9060debe139e.pkl"
CODING_ID_PATH = REPO_ROOT / "results/tables/cot_quality_features_with_neurons.csv"

MATH_META = {
    "aime24": NAD_ROOT / "MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/aime24/cache_neuron_output_1_act_no_rms_20250902_025610/meta.json",
    "aime25": NAD_ROOT / "MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/aime25/cache_neuron_output_1_act_no_rms_20251126_114548/meta.json",
    "brumo25": NAD_ROOT / "MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/brumo25/cache_neuron_output_1_act_no_rms_20251126_071142/meta.json",
    "hmmt25": NAD_ROOT / "MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/hmmt25/cache_neuron_output_1_act_no_rms_20251126_223151/meta.json",
}

MATH_RAW_JSONL = REPO_ROOT / "results/glm_math_knot_span_raw_v1_pilot64/all_runs.jsonl"
MATH_VERIFY_JSONL = REPO_ROOT / "results/glm_math_knot_span_verify_v1_pilot64/verified_spans.jsonl"

CODING_LABEL_PATHS = [
    REPO_ROOT / "results/tables/glm_nl_linearization_labels_partial.csv",
    REPO_ROOT / "results/tables/glm_nl_linearization_labels_partial96.csv",
    REPO_ROOT / "results/tables/glm_nl_linearization_labels_smoke.csv",
    REPO_ROOT / "results/tables/glm_nl_linearization_labels_smoke3.csv",
]

OUT_RUNS = REPO_ROOT / "results/tables/knot_cross_domain_run_table_pilot64.csv"
OUT_EFFECTS = REPO_ROOT / "results/tables/knot_cross_domain_feature_effects_pilot64.csv"
OUT_PROFILES = REPO_ROOT / "results/tables/knot_cross_domain_temporal_profiles_pilot64.csv"
OUT_PATTERNS = REPO_ROOT / "results/tables/knot_cross_domain_label_patterns_pilot64.csv"

ANCHOR_POS_TO_IDX = {10: 1, 40: 5, 70: 8, 100: 11}
ANCHORS = [10, 40, 70, 100]

RAW25_FEATURE_NAMES = list(FULL_FEATURE_NAMES[:25])
RAW25_FEATURE_TO_INDEX = {name: idx for idx, name in enumerate(RAW25_FEATURE_NAMES)}

SELECTED_FEATURES = [
    "tok_conf_prefix",
    "tok_conf_recency",
    "traj_reflection_count",
    "traj_novelty",
    "traj_continuity",
    "traj_max_reflection",
    "traj_late_convergence",
    "nc_mean",
    "nc_slope",
    "self_similarity",
]

PROFILE_FEATURES = [
    "tok_conf_prefix",
    "traj_reflection_count",
    "traj_novelty",
    "traj_continuity",
    "nc_mean",
    "self_similarity",
]


def cohen_d(correct_vals: np.ndarray, incorrect_vals: np.ndarray) -> float:
    correct_vals = np.asarray(correct_vals, dtype=float)
    incorrect_vals = np.asarray(incorrect_vals, dtype=float)
    if len(correct_vals) < 2 or len(incorrect_vals) < 2:
        return float("nan")
    mean_diff = float(correct_vals.mean() - incorrect_vals.mean())
    pooled_num = (len(correct_vals) - 1) * correct_vals.var(ddof=1) + (len(incorrect_vals) - 1) * incorrect_vals.var(ddof=1)
    pooled_den = len(correct_vals) + len(incorrect_vals) - 2
    if pooled_den <= 0:
        return float("nan")
    pooled_sd = math.sqrt(max(pooled_num / pooled_den, 1e-12))
    return mean_diff / pooled_sd


def mannwhitney_p(correct_vals: np.ndarray, incorrect_vals: np.ndarray) -> float:
    correct_vals = np.asarray(correct_vals, dtype=float)
    incorrect_vals = np.asarray(incorrect_vals, dtype=float)
    if len(correct_vals) == 0 or len(incorrect_vals) == 0:
        return float("nan")
    try:
        _, p_val = mannwhitneyu(correct_vals, incorrect_vals, alternative="two-sided")
        return float(p_val)
    except ValueError:
        return float("nan")


def load_math_meta_map() -> dict[str, dict[int, tuple[str, int]]]:
    meta_map: dict[str, dict[int, tuple[str, int]]] = {}
    for dataset, path in MATH_META.items():
        meta = json.loads(path.read_text())
        sample_map = {}
        for sample_id, sample in enumerate(meta.get("samples", [])):
            sample_map[int(sample_id)] = (str(sample["problem_id"]), int(sample["run_index"]))
        meta_map[dataset] = sample_map
    return meta_map


def build_cache_feature_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    meta_map = load_math_meta_map()
    bundle = pickle.load(open(CACHE_PATH, "rb"))

    math_rows = []
    coding_rows = []
    coding_ids = pd.read_csv(CODING_ID_PATH)[["rec_idx", "problem_id", "run_index", "is_correct"]]

    for store in bundle["feature_store"]:
        domain = str(store["domain"])
        dataset = str(store["dataset_name"])
        tensor = np.asarray(store["tensor"], dtype=float)
        labels = np.asarray(store["labels"], dtype=int)
        sample_ids = np.asarray(store["sample_ids"], dtype=int)

        for row_idx, sample_id in enumerate(sample_ids):
            row = {}
            for feature in SELECTED_FEATURES:
                f_idx = RAW25_FEATURE_TO_INDEX[feature]
                for anchor in ANCHORS:
                    pos_idx = ANCHOR_POS_TO_IDX[anchor]
                    row[f"{feature}_{anchor}"] = float(tensor[row_idx, pos_idx, f_idx])
                row[f"{feature}_delta_100_10"] = row[f"{feature}_100"] - row[f"{feature}_10"]
            if domain == "math":
                problem_id, run_index = meta_map[dataset][int(sample_id)]
                row.update({
                    "domain": "math",
                    "dataset": dataset,
                    "problem_id": problem_id,
                    "run_index": int(run_index),
                    "is_correct": int(labels[row_idx]),
                    "sample_id": int(sample_id),
                })
                math_rows.append(row)
            elif domain == "coding":
                row.update({
                    "domain": "coding",
                    "dataset": dataset,
                    "rec_idx": int(sample_id),
                    "is_correct_cache": int(labels[row_idx]),
                })
                coding_rows.append(row)

    math_df = pd.DataFrame(math_rows)
    coding_df = pd.DataFrame(coding_rows).merge(coding_ids, on="rec_idx", how="left", validate="one_to_one")
    label_mismatch = int((coding_df["is_correct_cache"] != coding_df["is_correct"]).sum())
    if label_mismatch:
        print(f"WARNING: coding cache/id label mismatches = {label_mismatch}")
    coding_df = coding_df.drop(columns=["is_correct_cache"])
    return math_df, coding_df


def load_math_run_labels() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_rows = []
    with open(MATH_RAW_JSONL) as handle:
        for line in handle:
            record = json.loads(line)
            parsed = record.get("glm_parsed", {}) or {}
            parse_error = record.get("parse_error")
            if parse_error or not isinstance(parsed.get("explicit_knot_spans", []), list):
                continue
            raw_rows.append({
                "domain": "math",
                "dataset": record["dataset"],
                "problem_id": str(record["problem_id"]),
                "run_index": int(record["run_index"]),
                "is_correct": int(record["is_correct"]),
                "n_candidate_spans": len(parsed.get("explicit_knot_spans", [])),
                "trace_strategy": str(parsed.get("trace_strategy", "")),
                "overall_state": str(parsed.get("overall_state", "")),
            })
    raw_df = pd.DataFrame(raw_rows)

    verify_rows = []
    with open(MATH_VERIFY_JSONL) as handle:
        for line in handle:
            record = json.loads(line)
            parsed = record.get("glm_parsed", {}) or {}
            verify_rows.append({
                "dataset": record["dataset"],
                "problem_id": str(record["problem_id"]),
                "run_index": int(record["run_index"]),
                "is_correct": int(record["is_correct"]),
                "span_idx": int(record["span_idx"]),
                "symptom": record["symptom"],
                "evidence_type": str(parsed.get("evidence_type", "")),
                "verdict": str(parsed.get("valid_explicit_knot", "")),
            })
    verify_df = pd.DataFrame(verify_rows)

    accepted = verify_df[verify_df["verdict"] == "yes"].copy()
    if accepted.empty:
        accepted_run = raw_df[["dataset", "problem_id", "run_index"]].copy()
        accepted_run["n_yes_spans"] = 0
        accepted_run["has_knot"] = 0
        accepted_run["math_evidence_types"] = ""
        accepted_run["math_symptoms"] = ""
    else:
        accepted_run = (
            accepted.groupby(["dataset", "problem_id", "run_index"])
            .agg(
                n_yes_spans=("span_idx", "count"),
                math_evidence_types=("evidence_type", lambda s: "|".join(sorted(set(str(x) for x in s if str(x))))),
                math_symptoms=("symptom", lambda s: "|".join(sorted(set(str(x) for x in s if str(x))))),
            )
            .reset_index()
        )
        accepted_run["has_knot"] = 1

    math_runs = raw_df.merge(accepted_run, on=["dataset", "problem_id", "run_index"], how="left")
    math_runs["n_yes_spans"] = math_runs["n_yes_spans"].fillna(0).astype(int)
    math_runs["has_knot"] = math_runs["has_knot"].fillna(0).astype(int)
    math_runs["math_evidence_types"] = math_runs["math_evidence_types"].fillna("")
    math_runs["math_symptoms"] = math_runs["math_symptoms"].fillna("")
    return math_runs, accepted


def _mode_or_first(series: pd.Series) -> str:
    nonempty = [str(x) for x in series if pd.notna(x) and str(x)]
    if not nonempty:
        return ""
    counts = pd.Series(nonempty).value_counts()
    return str(counts.index[0])


def load_coding_run_labels() -> pd.DataFrame:
    frames = []
    for path in CODING_LABEL_PATHS:
        df = pd.read_csv(path)
        df["source"] = path.name
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    agg = (
        all_df.groupby("rec_idx")
        .agg(
            problem_id=("problem_id", "first"),
            is_correct=("is_correct", "first"),
            problem_bucket=("problem_bucket", _mode_or_first),
            knot_present=("knot_present", _mode_or_first),
            knot_present_bin=("knot_present_bin", "max"),
            knot_severity=("knot_severity", "max"),
            primary_trigger=("primary_trigger", _mode_or_first),
            trace_strategy=("trace_strategy", _mode_or_first),
            reversal_count=("reversal_count", "max"),
            state_consistency=("state_consistency", _mode_or_first),
            coding_symptoms=("knot_symptoms", lambda s: "|".join(sorted(set(str(x) for x in s if pd.notna(x) and str(x))))),
        )
        .reset_index()
    )
    agg["domain"] = "coding"
    agg["dataset"] = "livecodebench_v5"
    agg["has_knot"] = agg["knot_present_bin"].astype(int)
    return agg


def build_run_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    math_feat, coding_feat = build_cache_feature_tables()
    math_labels, math_accepted_spans = load_math_run_labels()
    coding_labels = load_coding_run_labels()

    math_runs = math_labels.merge(
        math_feat,
        on=["domain", "dataset", "problem_id", "run_index", "is_correct"],
        how="inner",
        validate="one_to_one",
    )
    coding_runs = coding_labels.merge(
        coding_feat,
        on=["domain", "dataset", "problem_id", "is_correct", "rec_idx"],
        how="inner",
        validate="one_to_one",
    )
    run_table = pd.concat([math_runs, coding_runs], ignore_index=True, sort=False)
    return run_table, math_accepted_spans


def build_feature_effects(run_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    subset = run_table[run_table["has_knot"] == 1].copy()

    for feature in SELECTED_FEATURES:
        feature_col = f"{feature}_100"
        delta_col = f"{feature}_delta_100_10"
        for metric_col in [feature_col, delta_col]:
            row = {"feature_metric": metric_col}
            domain_effects = {}
            for domain in ["math", "coding"]:
                sub = subset[subset["domain"] == domain]
                correct = sub[sub["is_correct"] == 1][metric_col].dropna().values
                incorrect = sub[sub["is_correct"] == 0][metric_col].dropna().values
                row[f"{domain}_n_correct"] = len(correct)
                row[f"{domain}_n_incorrect"] = len(incorrect)
                row[f"{domain}_mean_correct"] = float(np.mean(correct)) if len(correct) else float("nan")
                row[f"{domain}_mean_incorrect"] = float(np.mean(incorrect)) if len(incorrect) else float("nan")
                row[f"{domain}_diff_correct_minus_incorrect"] = (
                    row[f"{domain}_mean_correct"] - row[f"{domain}_mean_incorrect"]
                    if len(correct) and len(incorrect) else float("nan")
                )
                row[f"{domain}_cohen_d"] = cohen_d(correct, incorrect)
                row[f"{domain}_mw_p"] = mannwhitney_p(correct, incorrect)
                domain_effects[domain] = row[f"{domain}_cohen_d"]
            row["domain_semantics_gap_cohen_d"] = (
                domain_effects["math"] - domain_effects["coding"]
                if not (math.isnan(domain_effects["math"]) or math.isnan(domain_effects["coding"]))
                else float("nan")
            )
            rows.append(row)
    effects = pd.DataFrame(rows)
    effects["abs_domain_semantics_gap"] = effects["domain_semantics_gap_cohen_d"].abs()
    return effects.sort_values("abs_domain_semantics_gap", ascending=False)


def build_temporal_profiles(run_table: pd.DataFrame) -> pd.DataFrame:
    subset = run_table[run_table["has_knot"] == 1].copy()
    rows = []
    for domain in ["math", "coding"]:
        for is_correct in [0, 1]:
            sub = subset[(subset["domain"] == domain) & (subset["is_correct"] == is_correct)]
            for feature in PROFILE_FEATURES:
                for anchor in ANCHORS:
                    col = f"{feature}_{anchor}"
                    rows.append({
                        "domain": domain,
                        "is_correct": is_correct,
                        "feature": feature,
                        "anchor_pct": anchor,
                        "n": int(sub[col].notna().sum()),
                        "mean_value": float(sub[col].mean()) if len(sub) else float("nan"),
                        "std_value": float(sub[col].std(ddof=1)) if len(sub) > 1 else float("nan"),
                    })
    return pd.DataFrame(rows)


def build_label_patterns(run_table: pd.DataFrame, math_accepted_spans: pd.DataFrame) -> pd.DataFrame:
    rows = []

    if not math_accepted_spans.empty:
        math_pattern = (
            math_accepted_spans.groupby(["evidence_type"])
            .size()
            .reset_index(name="n")
            .assign(domain="math", label_key="evidence_type", label_value=lambda df: df["evidence_type"], is_correct=np.nan)
        )[["domain", "label_key", "label_value", "is_correct", "n"]]
        rows.append(math_pattern)

        math_by_correct = (
            math_accepted_spans.groupby(["evidence_type", "verdict"])
            .size()
            .reset_index(name="n")
        )
        _ = math_by_correct  # quiet unused in future edits

    if not math_accepted_spans.empty:
        tmp = (
            math_accepted_spans[math_accepted_spans["verdict"] == "yes"]
            .groupby(["evidence_type", "is_correct"])
            .size()
            .reset_index(name="n")
            .rename(columns={"evidence_type": "label_value"})
        )
        if not tmp.empty:
            tmp["domain"] = "math"
            tmp["label_key"] = "evidence_type_by_correct"
            rows.append(tmp[["domain", "label_key", "label_value", "is_correct", "n"]])

    coding_yes = run_table[(run_table["domain"] == "coding") & (run_table["has_knot"] == 1)].copy()
    if not coding_yes.empty:
        for key in ["primary_trigger", "trace_strategy", "state_consistency"]:
            tmp = (
                coding_yes.groupby([key, "is_correct"])
                .size()
                .reset_index(name="n")
                .rename(columns={key: "label_value"})
            )
            tmp["domain"] = "coding"
            tmp["label_key"] = key
            rows.append(tmp[["domain", "label_key", "label_value", "is_correct", "n"]])

    if not rows:
        return pd.DataFrame(columns=["domain", "label_key", "label_value", "is_correct", "n"])
    return pd.concat(rows, ignore_index=True)


def main():
    run_table, math_accepted_spans = build_run_table()
    run_table.to_csv(OUT_RUNS, index=False)

    effects = build_feature_effects(run_table)
    effects.to_csv(OUT_EFFECTS, index=False)

    profiles = build_temporal_profiles(run_table)
    profiles.to_csv(OUT_PROFILES, index=False)

    patterns = build_label_patterns(run_table, math_accepted_spans)
    patterns.to_csv(OUT_PATTERNS, index=False)

    print(f"Saved run table   -> {OUT_RUNS}")
    print(f"Saved effects     -> {OUT_EFFECTS}")
    print(f"Saved profiles    -> {OUT_PROFILES}")
    print(f"Saved patterns    -> {OUT_PATTERNS}")

    knot_subset = run_table[run_table["has_knot"] == 1].copy()
    print("\n=== Knot-positive run counts ===")
    print(knot_subset.groupby(["domain", "is_correct"]).size().to_string())

    print("\n=== Top domain-semantics gaps ===")
    show_cols = [
        "feature_metric",
        "math_mean_correct",
        "math_mean_incorrect",
        "math_cohen_d",
        "coding_mean_correct",
        "coding_mean_incorrect",
        "coding_cohen_d",
        "domain_semantics_gap_cohen_d",
    ]
    print(effects[show_cols].head(12).to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
