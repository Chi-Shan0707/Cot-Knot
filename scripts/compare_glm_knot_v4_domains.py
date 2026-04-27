"""
Compare GLM knot labels v4 across domains
=========================================
Combine per-domain v4 tables into cross-domain summaries and a review scaffold.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from knot_glm_common import REPO_ROOT
from knot_v4_configs import DOMAIN_CONFIGS


TABLE_DIR = REPO_ROOT / "results" / "tables"


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def build_cross_summary(domains: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_frames = []
    symptom_frames = []
    label_frames = []
    for domain in domains:
        summary_path = TABLE_DIR / f"glm_{domain}_knot_summary_v4.csv"
        symptoms_path = TABLE_DIR / f"glm_{domain}_knot_symptoms_v4.csv"
        labels_path = TABLE_DIR / f"glm_{domain}_knot_labels_v4.csv"
        summary_frames.append(read_table(summary_path))
        symptom_frames.append(read_table(symptoms_path))
        label_frames.append(read_table(labels_path))
    cross_summary = pd.concat(summary_frames, ignore_index=True)
    cross_symptoms = pd.concat(symptom_frames, ignore_index=True)
    cross_labels = pd.concat(label_frames, ignore_index=True)
    return cross_summary, cross_symptoms, cross_labels


def build_audit_sample(labels: pd.DataFrame, audit_n: int = 30, seed: int = 42) -> pd.DataFrame:
    frames = []
    for domain, group in labels.groupby("domain"):
        valid = group[group["parse_ok"] == 1].copy()
        if valid.empty:
            continue
        positives = valid[valid["knot_present_bin"] == 1]
        negatives = valid[valid["knot_present_bin"] == 0]
        take_pos = min(len(positives), audit_n // 2)
        take_neg = min(len(negatives), audit_n - take_pos)
        sample = pd.concat(
            [
                positives.sample(n=take_pos, random_state=seed) if take_pos else positives.head(0),
                negatives.sample(n=take_neg, random_state=seed) if take_neg else negatives.head(0),
            ],
            ignore_index=True,
        )
        sample = sample.sort_values(["problem_key", "run_index"]).copy()
        sample["review_is_knot"] = ""
        sample["review_primary_symptom"] = ""
        sample["review_notes"] = ""
        frames.append(
            sample[
                [
                    "domain",
                    "dataset",
                    "problem_id",
                    "problem_key",
                    "run_index",
                    "is_correct",
                    "knot_present",
                    "knot_severity",
                    "state_consistency",
                    "primary_trigger",
                    "knot_symptoms",
                    "knot_quote",
                    "open_diagnosis",
                    "review_is_knot",
                    "review_primary_symptom",
                    "review_notes",
                ]
            ]
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_findings_markdown(summary: pd.DataFrame, symptoms: pd.DataFrame, out_path: Path):
    lines = [
        "# GLM Knot Findings v4",
        "",
        "This file summarizes the current v4 balanced-sample knot annotation outputs.",
        "",
        "## Cross-domain summary",
        "",
    ]
    for _, row in summary.sort_values("domain").iterrows():
        lines.append(
            f"- **{row['domain']}**: runs={int(row['total_runs'])}, "
            f"accuracy={row['pct_correct']:.3f}, prevalence={row['knot_prevalence']:.3f}, "
            f"rate_correct={row['knot_rate_correct']:.3f}, rate_incorrect={row['knot_rate_incorrect']:.3f}, "
            f"mean_severity={row['mean_knot_severity']:.3f}, "
            f"rho(knot,correct)={row['spearman_rho_knot_vs_correct']:.3f}, "
            f"rho(severity,correct)={row['spearman_rho_severity_vs_correct']:.3f}, "
            f"degenerate={int(row['degenerate_flag'])}"
        )
    lines.extend(["", "## Top symptoms by domain", ""])
    for domain in sorted(summary["domain"].unique()):
        lines.append(f"### {domain}")
        top = symptoms[symptoms["domain"] == domain].sort_values(["prevalence", "n_present"], ascending=False).head(3)
        for _, row in top.iterrows():
            lines.append(
                f"- `{row['symptom']}`: prevalence={row['prevalence']:.3f}, "
                f"rate_correct={row['rate_correct']:.3f}, rate_incorrect={row['rate_incorrect']:.3f}"
            )
        lines.append("")
    out_path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Compare GLM knot labels v4 across domains")
    parser.add_argument("--domains", nargs="+", default=["math", "science", "coding"], choices=sorted(DOMAIN_CONFIGS))
    parser.add_argument("--audit-n", type=int, default=30)
    args = parser.parse_args()

    cross_summary, cross_symptoms, cross_labels = build_cross_summary(args.domains)
    cross_summary_path = TABLE_DIR / "glm_knot_cross_domain_summary_v4.csv"
    cross_symptoms_path = TABLE_DIR / "glm_knot_cross_domain_symptoms_v4.csv"
    audit_path = TABLE_DIR / "glm_knot_protocol_audit_v4.csv"
    findings_path = TABLE_DIR / "glm_knot_findings_v4.md"

    cross_summary.to_csv(cross_summary_path, index=False)
    cross_symptoms.to_csv(cross_symptoms_path, index=False)
    audit_df = build_audit_sample(cross_labels, audit_n=args.audit_n)
    audit_df.to_csv(audit_path, index=False)
    write_findings_markdown(cross_summary, cross_symptoms, findings_path)

    print(f"Saved cross summary  -> {cross_summary_path}")
    print(f"Saved cross symptoms -> {cross_symptoms_path}")
    print(f"Saved audit sample   -> {audit_path}")
    print(f"Saved findings       -> {findings_path}")


if __name__ == "__main__":
    main()
