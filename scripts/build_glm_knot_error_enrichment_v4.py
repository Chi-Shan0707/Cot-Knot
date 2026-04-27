"""
Build GLM knot v4 error-enrichment tables
=========================================

Join final v4 knot labels to cache-derived confidence / entropy / neuron-summary
features so parse errors, filtered raw positives, and accepted severe knots can
be reviewed with aligned auxiliary signals.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from knot_glm_common import NAD_ROOT, REPO_ROOT
from knot_v4_configs import DOMAIN_CONFIGS


sys.path.insert(0, str(NAD_ROOT))
from nad.ops.earlystop_svd import LEGACY_FULL_FEATURE_NAMES  # noqa: E402


TABLE_DIR = REPO_ROOT / "results" / "tables"
CACHE_PATH = NAD_ROOT / "results" / "cache" / "es_svd_ms_rr_r1" / "cache_all_547b9060debe139e.pkl"
CODING_EXTRA_PATH = REPO_ROOT.parent.parent / "results" / "tables" / "nl_structure_features.csv"

ANCHOR_POS_TO_IDX = {10: 1, 40: 5, 70: 8, 100: 11}
ANCHORS = [10, 40, 70, 100]
FEATURE_NAMES = list(LEGACY_FULL_FEATURE_NAMES)
FEATURE_TO_INDEX = {name: idx for idx, name in enumerate(FEATURE_NAMES)}
SELECTED_FEATURES = [
    "tok_conf_prefix",
    "tok_conf_recency",
    "tok_neg_entropy_prefix",
    "tok_neg_entropy_recency",
    "tok_selfcert_prefix",
    "tok_selfcert_recency",
    "nc_mean",
    "nc_slope",
    "self_similarity",
]
SUMMARY_METRICS = [
    "tok_conf_prefix_100",
    "tok_conf_prefix_delta_100_10",
    "tok_neg_entropy_prefix_100",
    "tok_neg_entropy_prefix_delta_100_10",
    "tok_selfcert_prefix_100",
    "tok_selfcert_prefix_delta_100_10",
    "nc_mean_100",
    "nc_mean_delta_100_10",
    "nc_slope_100",
    "self_similarity_100",
]
CODING_EXTRA_COLUMNS = [
    "n_active_L19",
    "n_active_L21",
    "wmax_max_L5",
    "wmax_max_L10",
    "cert_density",
    "uncert_density",
    "cot_quality_score_plus_neuron",
]


def normalize_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["problem_id"] = out["problem_id"].astype(str)
    out["run_index"] = out["run_index"].astype(int)
    out["is_correct"] = out["is_correct"].astype(int)
    if "dataset" in out.columns:
        out["dataset"] = out["dataset"].astype(str)
    if "domain" in out.columns:
        out["domain"] = out["domain"].astype(str)
    return out


def load_label_tables(domains: list[str]) -> pd.DataFrame:
    frames = []
    for domain in domains:
        path = TABLE_DIR / f"glm_{domain}_knot_labels_v4.csv"
        df = pd.read_csv(path)
        df = normalize_id_columns(df)
        frames.append(df)
    labels = pd.concat(frames, ignore_index=True, sort=False)
    labels["raw_knot_present"] = labels["raw_knot_present"].fillna("").astype(str)
    labels["protocol_filter"] = labels["protocol_filter"].fillna("").astype(str)
    return labels


def dataset_meta_paths() -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for config in DOMAIN_CONFIGS.values():
        for dataset, report_path in config.report_paths.items():
            meta_path = report_path.parent / "meta.json"
            paths[dataset] = meta_path
    return paths


def load_sample_maps() -> dict[str, dict[int, tuple[str, int]]]:
    sample_maps: dict[str, dict[int, tuple[str, int]]] = {}
    for dataset, meta_path in dataset_meta_paths().items():
        meta = json.loads(meta_path.read_text())
        sample_map: dict[int, tuple[str, int]] = {}
        for sample_id, sample in enumerate(meta.get("samples", [])):
            sample_map[int(sample_id)] = (str(sample["problem_id"]), int(sample["run_index"]))
        sample_maps[dataset] = sample_map
    return sample_maps


def build_cache_feature_table(domains: list[str]) -> pd.DataFrame:
    sample_maps = load_sample_maps()
    bundle = pickle.load(CACHE_PATH.open("rb"))
    rows = []

    for store in bundle["feature_store"]:
        domain = str(store["domain"])
        dataset = str(store["dataset_name"])
        if domain not in domains:
            continue
        tensor = np.asarray(store["tensor"], dtype=float)
        labels = np.asarray(store["labels"], dtype=int)
        sample_ids = np.asarray(store["sample_ids"], dtype=int)
        sample_map = sample_maps[dataset]

        for row_idx, sample_id in enumerate(sample_ids):
            problem_id, run_index = sample_map[int(sample_id)]
            row = {
                "domain": domain,
                "dataset": dataset,
                "problem_id": str(problem_id),
                "run_index": int(run_index),
                "is_correct_cache": int(labels[row_idx]),
                "sample_id": int(sample_id),
            }
            for feature in SELECTED_FEATURES:
                feat_idx = FEATURE_TO_INDEX[feature]
                for anchor in ANCHORS:
                    pos_idx = ANCHOR_POS_TO_IDX[anchor]
                    row[f"{feature}_{anchor}"] = float(tensor[row_idx, pos_idx, feat_idx])
                row[f"{feature}_delta_100_10"] = row[f"{feature}_100"] - row[f"{feature}_10"]
            rows.append(row)

    feature_df = pd.DataFrame(rows)
    feature_df["problem_id"] = feature_df["problem_id"].astype(str)
    feature_df["run_index"] = feature_df["run_index"].astype(int)
    feature_df["is_correct_cache"] = feature_df["is_correct_cache"].astype(int)
    feature_df["domain"] = feature_df["domain"].astype(str)
    feature_df["dataset"] = feature_df["dataset"].astype(str)
    return feature_df


def load_coding_extra_features() -> pd.DataFrame:
    if not CODING_EXTRA_PATH.exists():
        return pd.DataFrame(columns=["problem_id", "run_index", "is_correct", *CODING_EXTRA_COLUMNS])
    df = pd.read_csv(CODING_EXTRA_PATH)
    keep = ["problem_id", "run_index", "is_correct", *CODING_EXTRA_COLUMNS]
    keep = [column for column in keep if column in df.columns]
    df = df[keep].drop_duplicates(subset=["problem_id", "run_index", "is_correct"])
    df["problem_id"] = df["problem_id"].astype(str)
    df["run_index"] = df["run_index"].astype(int)
    df["is_correct"] = df["is_correct"].astype(int)
    return df


def assign_error_bucket(row: pd.Series) -> str:
    if int(row["parse_ok"]) == 0:
        return "parse_error"
    if str(row["raw_knot_present"]) == "yes" and int(row["knot_present_bin"]) == 0:
        return "filtered_raw_positive"
    if int(row["knot_present_bin"]) == 1 and int(row["knot_severity"]) >= 2:
        return "accepted_severe_knot"
    if int(row["knot_present_bin"]) == 1:
        return "accepted_mild_knot"
    return "clean_negative"


def enrich_labels(labels: pd.DataFrame, cache_features: pd.DataFrame, coding_extra: pd.DataFrame) -> pd.DataFrame:
    merged = labels.merge(
        cache_features,
        on=["domain", "dataset", "problem_id", "run_index"],
        how="left",
        validate="one_to_one",
        indicator="cache_merge_status",
    )
    merged["cache_join_ok"] = (merged["cache_merge_status"] == "both").astype(int)
    merged = merged.drop(columns=["cache_merge_status"])
    merged["cache_correctness_match"] = np.where(
        merged["cache_join_ok"] == 1,
        (merged["is_correct"] == merged["is_correct_cache"]).astype(int),
        np.nan,
    )

    coding_mask = merged["domain"] == "coding"
    if coding_mask.any() and not coding_extra.empty:
        coding_rows = merged.loc[coding_mask].merge(
            coding_extra,
            on=["problem_id", "run_index"],
            how="left",
            validate="one_to_one",
            indicator="coding_extra_merge_status",
            suffixes=("", "_coding_extra"),
        )
        coding_rows["coding_extra_join_ok"] = (coding_rows["coding_extra_merge_status"] == "both").astype(int)
        coding_rows = coding_rows.drop(columns=["coding_extra_merge_status"])
        if "is_correct_coding_extra" in coding_rows.columns:
            coding_rows["coding_extra_correctness_match"] = np.where(
                coding_rows["coding_extra_join_ok"] == 1,
                (coding_rows["is_correct"] == coding_rows["is_correct_coding_extra"]).astype(int),
                np.nan,
            )
            coding_rows = coding_rows.drop(columns=["is_correct_coding_extra"])
        merged = pd.concat([merged.loc[~coding_mask], coding_rows], ignore_index=True, sort=False)
    else:
        merged["coding_extra_join_ok"] = 0
        merged["coding_extra_correctness_match"] = np.nan
        for column in CODING_EXTRA_COLUMNS:
            merged[column] = np.nan

    merged["error_bucket"] = merged.apply(assign_error_bucket, axis=1)
    merged["error_focus"] = merged["error_bucket"].isin(
        {"parse_error", "filtered_raw_positive", "accepted_severe_knot"}
    ).astype(int)
    merged["raw_positive_bin"] = (merged["raw_knot_present"] == "yes").astype(int)
    merged["problem_id"] = merged["problem_id"].astype(str)
    merged["problem_key"] = merged["problem_key"].astype(str)
    return merged.sort_values(["domain", "dataset", "problem_id", "run_index"]).reset_index(drop=True)


def summarize_enrichment(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (domain, error_bucket), group in df.groupby(["domain", "error_bucket"], sort=True):
        row = {
            "domain": domain,
            "error_bucket": error_bucket,
            "n_runs": int(len(group)),
            "pct_correct": float(group["is_correct"].mean()) if len(group) else np.nan,
            "parse_ok_rate": float(group["parse_ok"].mean()) if len(group) else np.nan,
            "raw_positive_rate": float(group["raw_positive_bin"].mean()) if len(group) else np.nan,
            "final_positive_rate": float(group["knot_present_bin"].mean()) if len(group) else np.nan,
            "mean_knot_severity": float(group["knot_severity"].mean()) if len(group) else np.nan,
            "mean_raw_reversal_count": float(pd.to_numeric(group["raw_reversal_count"], errors="coerce").mean()),
            "cache_join_rate": float(group["cache_join_ok"].mean()) if len(group) else np.nan,
            "cache_correctness_match_rate": float(pd.to_numeric(group["cache_correctness_match"], errors="coerce").mean()),
            "coding_extra_join_rate": float(group["coding_extra_join_ok"].mean()) if len(group) else np.nan,
            "coding_extra_correctness_match_rate": float(
                pd.to_numeric(group["coding_extra_correctness_match"], errors="coerce").mean()
            ),
        }
        for metric in SUMMARY_METRICS:
            row[f"mean_{metric}"] = float(pd.to_numeric(group[metric], errors="coerce").mean())
        for metric in CODING_EXTRA_COLUMNS:
            if metric in group.columns:
                row[f"mean_{metric}"] = float(pd.to_numeric(group[metric], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["domain", "error_bucket"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Build GLM knot v4 error enrichment tables")
    parser.add_argument("--domains", nargs="+", default=["math", "science", "coding"], choices=sorted(DOMAIN_CONFIGS))
    args = parser.parse_args()

    labels = load_label_tables(args.domains)
    cache_features = build_cache_feature_table(args.domains)
    coding_extra = load_coding_extra_features()
    enriched = enrich_labels(labels, cache_features, coding_extra)
    summary = summarize_enrichment(enriched)

    out_path = TABLE_DIR / "glm_knot_error_enrichment_v4.csv"
    summary_path = TABLE_DIR / "glm_knot_error_summary_v4.csv"
    enriched.to_csv(out_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Saved enrichment -> {out_path}")
    print(f"Saved summary    -> {summary_path}")
    print("\n=== Join coverage ===")
    print(
        enriched.groupby("domain")[["cache_join_ok", "coding_extra_join_ok"]]
        .mean()
        .reset_index()
        .to_string(index=False)
    )
    print("\n=== Error-focus counts ===")
    print(
        enriched.groupby(["domain", "error_bucket"])
        .size()
        .reset_index(name="n_runs")
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
