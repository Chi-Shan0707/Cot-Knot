"""
Experiment: add v4 knot labels as features for correctness prediction
=====================================================================

This script evaluates whether the current v4 knot annotations improve
run-level correctness discrimination and per-problem reranking on the
annotated subset.

Important scope note:
- The available v4 knot labels cover 64 problems × 4 runs/problem per domain.
- Therefore the downstream reranking metric here is best-of-4 on the annotated
  subset, not the full best-of-64 setting used elsewhere in the repo.
- This is still useful as an "oracle knot feature" upper-bound experiment:
  if even gold/GLM knot labels do not help, there is little reason to deploy
  a knot proxy.
- By default, rows with `parse_ok=0` are excluded because they do not carry a
  valid knot label. Use `--include-parse-errors` to keep them.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from knot_glm_common import NAD_ROOT, REPO_ROOT
from knot_v4_configs import DOMAIN_CONFIGS


sys.path.insert(0, str(NAD_ROOT))


TABLE_DIR = REPO_ROOT / "results" / "tables"
ENRICH_PATH = TABLE_DIR / "glm_knot_error_enrichment_v4.csv"
CACHE_PATH = NAD_ROOT / "results" / "cache" / "es_svd_ms_rr_r1" / "cache_all_547b9060debe139e.pkl"

ANCHOR_IDX = 11
FIXED_FEATURE_NAMES = [
    "tok_conf_prefix",
    "tok_conf_recency",
    "tok_gini_prefix",
    "tok_gini_tail",
    "tok_gini_slope",
    "tok_neg_entropy_prefix",
    "tok_neg_entropy_recency",
    "tok_selfcert_prefix",
    "tok_selfcert_recency",
    "tok_logprob_prefix",
    "tok_logprob_recency",
    "traj_continuity",
    "traj_reflection_count",
    "traj_novelty",
    "traj_max_reflection",
    "traj_late_convergence",
    "has_tok_conf",
    "has_tok_gini",
    "has_tok_neg_entropy",
    "has_tok_selfcert",
    "has_tok_logprob",
    "has_rows_bank",
]
FIXED_FEATURE_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 19, 20, 21, 22, 23, 24]
N_SPLITS = 5
RANDOM_STATE = 42


@dataclass(frozen=True)
class VariantSpec:
    name: str
    knot_cols: tuple[str, ...]


VARIANTS = [
    VariantSpec("base_only", ()),
    VariantSpec("base_plus_knot_bin", ("knot_present_bin",)),
    VariantSpec("base_plus_knot_severity", ("knot_severity",)),
    VariantSpec("base_plus_knot_rich", ("knot_present_bin", "knot_severity", "knot_count")),
]


def normalize_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["domain"] = out["domain"].astype(str)
    out["dataset"] = out["dataset"].astype(str)
    out["problem_id"] = out["problem_id"].astype(str)
    out["run_index"] = out["run_index"].astype(int)
    out["is_correct"] = out["is_correct"].astype(int)
    out["problem_key"] = out["problem_key"].astype(str)
    return out


def dataset_meta_paths() -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for config in DOMAIN_CONFIGS.values():
        for dataset, report_path in config.report_paths.items():
            paths[dataset] = report_path.parent / "meta.json"
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


def rank_transform_block(x_block: np.ndarray) -> np.ndarray:
    if x_block.shape[0] <= 1:
        return np.zeros_like(x_block, dtype=float)
    out = np.zeros_like(x_block, dtype=float)
    denom = max(x_block.shape[0] - 1, 1)
    for col_idx in range(x_block.shape[1]):
        order = np.argsort(x_block[:, col_idx], kind="mergesort")
        out[order, col_idx] = np.arange(len(order), dtype=float) / denom
    return out


def build_full_feature_table(domains: list[str]) -> pd.DataFrame:
    payload = pickle.load(CACHE_PATH.open("rb"))
    sample_maps = load_sample_maps()
    rows: list[dict] = []

    for store in payload["feature_store"]:
        domain = str(store["domain"])
        dataset = str(store["dataset_name"])
        if domain not in domains:
            continue
        tensor = np.asarray(store["tensor"], dtype=float)
        sample_ids = np.asarray(store["sample_ids"], dtype=int)
        problem_offsets = [int(v) for v in store["problem_offsets"]]
        if len(problem_offsets) == 0 or problem_offsets[-1] != tensor.shape[0]:
            problem_offsets = problem_offsets + [tensor.shape[0]]
        raw_anchor = tensor[:, ANCHOR_IDX, :][:, FIXED_FEATURE_INDICES]
        rank_anchor = np.zeros_like(raw_anchor)
        for start, end in zip(problem_offsets[:-1], problem_offsets[1:]):
            rank_anchor[start:end] = rank_transform_block(raw_anchor[start:end])

        sample_map = sample_maps[dataset]
        for row_idx, sample_id in enumerate(sample_ids):
            problem_id, run_index = sample_map[int(sample_id)]
            row = {
                "domain": domain,
                "dataset": dataset,
                "problem_id": str(problem_id),
                "run_index": int(run_index),
                "sample_id": int(sample_id),
            }
            for feat_idx, feat_name in enumerate(FIXED_FEATURE_NAMES):
                row[f"raw_{feat_name}"] = float(raw_anchor[row_idx, feat_idx])
                row[f"rank_{feat_name}"] = float(rank_anchor[row_idx, feat_idx])
            rows.append(row)

    return pd.DataFrame(rows)


def load_annotated_table(domains: list[str]) -> pd.DataFrame:
    df = pd.read_csv(ENRICH_PATH)
    df = normalize_key_columns(df)
    df = df[df["domain"].isin(domains)].copy()
    keep_cols = [
        "domain",
        "dataset",
        "problem_id",
        "problem_key",
        "run_index",
        "is_correct",
        "parse_ok",
        "knot_present_bin",
        "knot_severity",
        "knot_count",
        "knot_present",
        "protocol_filter",
        "sample_id",
    ]
    return df[keep_cols].copy()


def merge_features(labels: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    merged = labels.merge(
        features,
        on=["domain", "dataset", "problem_id", "run_index", "sample_id"],
        how="inner",
        validate="one_to_one",
    )
    return merged.sort_values(["domain", "dataset", "problem_id", "run_index"]).reset_index(drop=True)


def build_feature_matrix(df: pd.DataFrame, knot_cols: tuple[str, ...]) -> tuple[np.ndarray, list[str]]:
    base_cols = [f"raw_{name}" for name in FIXED_FEATURE_NAMES] + [f"rank_{name}" for name in FIXED_FEATURE_NAMES]
    all_cols = base_cols + list(knot_cols)
    x = df[all_cols].to_numpy(dtype=float, copy=True)
    return x, all_cols


def best_of_k(df: pd.DataFrame, score_col: str) -> float:
    picks = df.loc[df.groupby("problem_key")[score_col].idxmax()]
    return float(picks["is_correct"].mean())


def random_baseline(df: pd.DataFrame) -> float:
    return float(df.groupby("problem_key")["is_correct"].mean().mean())


def fit_oof_scores(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    n_splits = min(N_SPLITS, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.full(len(y), np.nan, dtype=float)

    for train_idx, val_idx in gkf.split(x, y, groups):
        n_comp = max(2, min(16, x.shape[1] - 1, len(train_idx) - 1))
        pipe = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("svd", TruncatedSVD(n_components=n_comp, random_state=RANDOM_STATE)),
                ("lr", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
            ]
        )
        pipe.fit(x[train_idx], y[train_idx])
        oof[val_idx] = pipe.predict_proba(x[val_idx])[:, 1]

    if np.isnan(oof).any():
        raise ValueError("OOF scores contain NaNs")
    return oof


def eval_variant(df: pd.DataFrame, variant: VariantSpec) -> dict:
    x, feature_cols = build_feature_matrix(df, variant.knot_cols)
    y = df["is_correct"].to_numpy(dtype=int)
    groups = df["problem_key"].to_numpy(dtype=str)
    oof = fit_oof_scores(x, y, groups)
    auc = float(roc_auc_score(y, oof)) if len(np.unique(y)) >= 2 else float("nan")
    work = df.copy()
    work["oof_score"] = oof
    best = best_of_k(work, "oof_score")
    rand = random_baseline(work)
    return {
        "variant": variant.name,
        "n_runs": int(len(work)),
        "n_problems": int(work["problem_key"].nunique()),
        "mean_runs_per_problem": float(work.groupby("problem_key").size().mean()),
        "auc_oof": auc,
        "bestof4_pass1_oof": best,
        "random_pass1": rand,
        "lift_vs_random": best - rand,
        "n_features": int(len(feature_cols)),
    }


def run_domain(df: pd.DataFrame, domain: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = df[df["domain"] == domain].copy()
    if sub.empty:
        return pd.DataFrame(), pd.DataFrame()
    summary_rows = []
    score_rows = []
    for variant in VARIANTS:
        metrics = eval_variant(sub, variant)
        metrics["domain"] = domain
        summary_rows.append(metrics)

        x, _ = build_feature_matrix(sub, variant.knot_cols)
        y = sub["is_correct"].to_numpy(dtype=int)
        groups = sub["problem_key"].to_numpy(dtype=str)
        oof = fit_oof_scores(x, y, groups)
        tmp = sub[["domain", "dataset", "problem_id", "problem_key", "run_index", "is_correct"]].copy()
        tmp["variant"] = variant.name
        tmp["oof_score"] = oof
        score_rows.append(tmp)
    return pd.DataFrame(summary_rows), pd.concat(score_rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate knot labels as extra features on annotated subset")
    parser.add_argument("--domains", nargs="+", default=["math", "science", "coding"], choices=sorted(DOMAIN_CONFIGS))
    parser.add_argument("--include-parse-errors", action="store_true")
    args = parser.parse_args()

    labels = load_annotated_table(args.domains)
    if not args.include_parse_errors:
        labels = labels[labels["parse_ok"] == 1].copy()
    features = build_full_feature_table(args.domains)
    merged = merge_features(labels, features)

    summary_frames = []
    score_frames = []
    for domain in args.domains:
        summary_df, score_df = run_domain(merged, domain)
        if not summary_df.empty:
            summary_frames.append(summary_df)
            score_frames.append(score_df)

    summary = pd.concat(summary_frames, ignore_index=True)
    scores = pd.concat(score_frames, ignore_index=True)
    summary["delta_auc_vs_base"] = summary["auc_oof"] - summary.groupby("domain")["auc_oof"].transform("first")
    summary["delta_bestof4_vs_base"] = summary["bestof4_pass1_oof"] - summary.groupby("domain")["bestof4_pass1_oof"].transform("first")

    out_summary = TABLE_DIR / "glm_knot_feature_gain_summary_v1.csv"
    out_scores = TABLE_DIR / "glm_knot_feature_gain_scores_v1.csv"
    summary.to_csv(out_summary, index=False)
    scores.to_csv(out_scores, index=False)

    print(f"Saved summary -> {out_summary}")
    print(f"Saved scores  -> {out_scores}")
    print("\n=== Summary ===")
    print(
        summary[
            [
                "domain",
                "variant",
                "auc_oof",
                "delta_auc_vs_base",
                "bestof4_pass1_oof",
                "random_pass1",
                "lift_vs_random",
                "delta_bestof4_vs_base",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
