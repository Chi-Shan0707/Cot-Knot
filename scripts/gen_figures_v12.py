#!/usr/bin/env python3
"""
Generate all figures for paper_v12.tex.

Run from any directory:
    python scripts/gen_figures_v12.py

Outputs (all PDF in workshop/cotknot/figures/):
    fig_auroc_by_anchor_v12.pdf   — AUROC curves by domain × anchor (Fig 1)
    fig_ssl_ceiling_v12.pdf        — SSL ceiling in coding (Fig 2)
    fig_reranking_v12.pdf          — Best-of-N reranking pass@1 (Fig 3)
    fig_deknot_alldomains_v12.pdf  — De-knotting ablation (Fig 4)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent          # workshop/cotknot
TABLES = ROOT / "results" / "tables"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SSL_CSV     = Path("/home/jovyan/work/SVDomain/results/tables/domain_specific_ssl_v2.csv")
RERANK_CSV  = Path("/home/jovyan/work/SVDomain/results/tables/bon_reranking_domain_pass1_ci.csv")
DEKNOT_CSV  = TABLES / "deknot_alldomains_v2.csv"

# ── Shared style ───────────────────────────────────────────────────────────────

DOMAIN_COLOR = {"math": "#1f77b4", "science": "#2ca02c", "coding": "#d62728"}
DOMAIN_LABEL = {"math": "Math", "science": "Science", "coding": "Coding"}
ANCHOR_TICKS = [10, 40, 70, 100]

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 200,
})


# ── Fig 1: AUROC by domain × anchor ──────────────────────────────────────────

def fig_auroc_by_anchor():
    """
    Left panel: AUROC at 4 anchor positions for each domain (no_svd_lr, lf=1.0).
    Right panel: AoA bar chart with bootstrap 95% CI.
    """
    ssl_df = pd.read_csv(SSL_CSV)
    sub = (ssl_df[(ssl_df.condition == "no_svd_lr") & (ssl_df.label_fraction == 1.0)]
           .groupby(["domain", "anchor_pct"])["auroc"].mean().reset_index())
    sub["anchor_pct"] = (sub["anchor_pct"] * 100).round().astype(int)

    # AoA values and CIs from paper (verified against bootstrap)
    aoa_data = {
        "math":    {"aoa": 0.958, "lo": 0.931, "hi": 0.980},
        "science": {"aoa": 0.799, "lo": 0.775, "hi": 0.822},
        "coding":  {"aoa": 0.434, "lo": 0.404, "hi": 0.464},
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)

    # Panel A: AUROC curves
    ax = axes[0]
    for domain in ["math", "science", "coding"]:
        rows = sub[sub.domain == domain].sort_values("anchor_pct")
        ax.plot(rows["anchor_pct"], rows["auroc"],
                marker="o", markersize=7, linewidth=2.4,
                color=DOMAIN_COLOR[domain], label=DOMAIN_LABEL[domain])
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.2, alpha=0.7, label="Random (0.5)")
    ax.set_xticks(ANCHOR_TICKS)
    ax.set_xlabel("Trace anchor (%)")
    ax.set_ylabel("AUROC")
    ax.set_title("(a) AUROC vs. anchor position")
    ax.set_ylim(0.35, 1.02)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.25)

    # Panel B: AoA bar chart with CI
    ax2 = axes[1]
    domains = ["math", "science", "coding"]
    xs = np.arange(len(domains))
    aoas = [aoa_data[d]["aoa"] for d in domains]
    errs_lo = [aoa_data[d]["aoa"] - aoa_data[d]["lo"] for d in domains]
    errs_hi = [aoa_data[d]["hi"] - aoa_data[d]["aoa"] for d in domains]
    colors = [DOMAIN_COLOR[d] for d in domains]
    bars = ax2.bar(xs, aoas, color=colors, width=0.5,
                   yerr=[errs_lo, errs_hi], capsize=5, error_kw={"linewidth": 2})
    ax2.axhline(0.5, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([DOMAIN_LABEL[d] for d in domains])
    ax2.set_ylabel("AUC-of-AUROC (AoA)")
    ax2.set_title("(b) AoA with 95% bootstrap CI")
    ax2.set_ylim(0.35, 1.05)
    for i, (x, v) in enumerate(zip(xs, aoas)):
        ax2.text(x, v + 0.025, f"{v:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax2.grid(alpha=0.25, axis="y")

    fig.savefig(FIG_DIR / "fig_auroc_by_anchor_v12.pdf", bbox_inches="tight")
    print("  Saved fig_auroc_by_anchor_v12.pdf")


# ── Fig 2: SSL ceiling ─────────────────────────────────────────────────────────

def fig_ssl_ceiling():
    """
    Left: Coding AUROC vs label fraction for all SSL conditions + baselines.
         Flat ceiling near 0.5 for all methods.
    Right: Same for Science to show contrast (SSL helps science).
    Use anchor=100% (anchor_pct=1.0).
    """
    ssl_df = pd.read_csv(SSL_CSV)
    # Focus on anchor=100% (full trace)
    sub = (ssl_df[ssl_df.anchor_pct == 1.0]
           .groupby(["domain", "condition", "label_fraction"])["auroc"].mean().reset_index())

    lf_pcts = sorted(sub.label_fraction.unique())
    lf_labels = [f"{int(lf*100)}%" for lf in lf_pcts]

    conditions = {
        "no_svd_lr":   ("StandardScaler+LR (no dim. red.)", "-",  "o", "#555555"),
        "frozen_svd":  ("Frozen supervised SVD+LR",         "--", "s", "#888888"),
        "ssl_v2_r8":   ("SSL v2 r=8 (domain-specific)",     "-",  "^", "#ff7f0e"),
        "ssl_v2_r16":  ("SSL v2 r=16",                      "-",  "D", "#e377c2"),
        "ssl_v2_r24":  ("SSL v2 r=24 (best SSL)",           "-",  "v", "#9467bd"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)

    for ax, domain, title in [
        (axes[0], "coding",  "(a) Coding — all methods near chance"),
        (axes[1], "science", "(b) Science — SSL helps at low labels"),
    ]:
        dom_sub = sub[sub.domain == domain]
        for cond, (label, ls, marker, color) in conditions.items():
            rows = dom_sub[dom_sub.condition == cond].sort_values("label_fraction")
            if len(rows) == 0:
                continue
            ax.plot(range(len(rows)), rows["auroc"].values,
                    linestyle=ls, marker=marker, markersize=7, linewidth=2.2,
                    color=color, label=label)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.5, alpha=0.7, label="Random (0.5)")
        ax.set_xticks(range(len(lf_pcts)))
        ax.set_xticklabels(lf_labels, rotation=30)
        ax.set_xlabel("Label fraction")
        ax.set_ylabel("AUROC at 100% anchor")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        if domain == "coding":
            ax.set_ylim(0.35, 0.65)
        else:
            ax.set_ylim(0.50, 0.85)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.12))

    fig.savefig(FIG_DIR / "fig_ssl_ceiling_v12.pdf", bbox_inches="tight")
    print("  Saved fig_ssl_ceiling_v12.pdf")


# ── Fig 3: Reranking pass@1 ────────────────────────────────────────────────────

def fig_reranking():
    """
    Grouped bar chart: Random vs Probe pass@1 per domain, with CI error bars.
    """
    df = pd.read_csv(RERANK_CSV)
    # Columns: domain, n_problems, n_candidates, baseline, baseline_ci_lo, baseline_ci_hi,
    #          probe, probe_ci_lo, probe_ci_hi, delta

    domains = df["domain"].tolist()
    n = len(domains)
    x = np.arange(n)
    w = 0.32

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)

    # Random baseline
    base = df["baseline"].values
    base_lo = df["baseline"] - df["baseline_ci_lo"].values
    base_hi = df["baseline_ci_hi"].values - df["baseline"].values
    bars1 = ax.bar(x - w/2, base, w, color="#aec7e8", label="Random (expected pass@1)",
                   yerr=[base_lo, base_hi], capsize=5, error_kw={"linewidth": 2})

    # Probe selection
    probe = df["probe"].values
    probe_lo = df["probe"] - df["probe_ci_lo"].values
    probe_hi = df["probe_ci_hi"].values - df["probe"].values
    bars2 = ax.bar(x + w/2, probe, w, color=[DOMAIN_COLOR[d] for d in domains],
                   label="Probe reranking (N=64)",
                   yerr=[probe_lo, probe_hi], capsize=5, error_kw={"linewidth": 2})

    # Delta annotations
    for i, (d, row) in enumerate(zip(domains, df.itertuples())):
        delta = row.delta
        color = "#2ca02c" if delta > 0 else ("#d62728" if delta < -0.01 else "#888888")
        sign = "+" if delta > 0 else ""
        ax.text(i, max(row.probe, row.baseline) + 0.055,
                f"{sign}{delta:.1%}", ha="center", va="bottom",
                fontsize=13, fontweight="bold", color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([DOMAIN_LABEL[d] for d in domains], fontsize=14)
    ax.set_ylabel("Pass@1", fontsize=13)
    ax.set_title("Best-of-N reranking (N = 64 candidates per problem)", fontsize=14)
    ax.set_ylim(0.45, 0.90)
    ax.legend(frameon=False, fontsize=12)
    ax.grid(alpha=0.25, axis="y")

    fig.savefig(FIG_DIR / "fig_reranking_v12.pdf", bbox_inches="tight")
    print("  Saved fig_reranking_v12.pdf")


# ── Fig 4: De-knotting ablation ────────────────────────────────────────────────

def fig_deknot():
    """
    Left panel: knot prevalence in correct vs incorrect runs per domain.
    Right panel: AUROC before vs after de-knotting per domain.
    Bottom: AUROC delta annotation.
    """
    df = pd.read_csv(DEKNOT_CSV)
    # Columns: domain, n_runs, n_knotted, knot_rate_correct, knot_rate_incorrect,
    #          total_chars_removed, auroc_original, auroc_deknot, delta, verdict

    domains = df["domain"].tolist()
    n = len(domains)
    x = np.arange(n)
    w = 0.32

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    # Panel A: Prevalence by outcome
    ax = axes[0]
    correct_rates   = df["knot_rate_correct"].values
    incorrect_rates = df["knot_rate_incorrect"].values
    bars_c = ax.bar(x - w/2, correct_rates,   w, color="#aec7e8", label="Correct runs")
    bars_i = ax.bar(x + w/2, incorrect_rates, w, color="#ffbb78", label="Incorrect runs")

    for i, (d, c, ic) in enumerate(zip(domains, correct_rates, incorrect_rates)):
        ax.text(i - w/2, c + 0.015,   f"{c:.0%}",  ha="center", va="bottom", fontsize=11)
        ax.text(i + w/2, ic + 0.015,  f"{ic:.0%}", ha="center", va="bottom", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels([DOMAIN_LABEL[d] for d in domains], fontsize=14)
    ax.set_ylabel("Fraction of runs with ≥1 knot", fontsize=13)
    ax.set_title("(a) Knot prevalence by outcome", fontsize=14)
    ax.set_ylim(0, 1.12)
    ax.legend(frameon=False, fontsize=12)
    ax.grid(alpha=0.25, axis="y")

    # Panel B: AUROC before vs after
    ax2 = axes[1]
    orig   = df["auroc_original"].values
    deknot = df["auroc_deknot"].values
    deltas = df["delta"].values

    bars_o = ax2.bar(x - w/2, orig,   w, color=[DOMAIN_COLOR[d] for d in domains],
                     alpha=0.85, label="Original")
    bars_d = ax2.bar(x + w/2, deknot, w, color=[DOMAIN_COLOR[d] for d in domains],
                     alpha=0.45, hatch="///", label="De-knotted")
    ax2.axhline(0.5, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)

    for i, (d, delta) in enumerate(zip(domains, deltas)):
        sign  = "+" if delta > 0 else ""
        color = "#d62728" if delta < -0.02 else ("#2ca02c" if delta > 0.02 else "#555555")
        ax2.text(i, max(orig[i], deknot[i]) + 0.022,
                 f"Δ={sign}{delta:.3f}", ha="center", va="bottom",
                 fontsize=12, fontweight="bold", color=color)

    ax2.set_xticks(x)
    ax2.set_xticklabels([DOMAIN_LABEL[d] for d in domains], fontsize=14)
    ax2.set_ylabel("AUROC (4-fold GroupKFold)", fontsize=13)
    ax2.set_title("(b) AUROC before vs. after knot-token removal", fontsize=14)
    ax2.set_ylim(0.35, 0.68)
    ax2.legend(frameon=False, fontsize=12, loc="upper right")
    ax2.grid(alpha=0.25, axis="y")

    fig.savefig(FIG_DIR / "fig_deknot_alldomains_v12.pdf", bbox_inches="tight")
    print("  Saved fig_deknot_alldomains_v12.pdf")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating figures for paper_v12...")
    fig_auroc_by_anchor()
    fig_ssl_ceiling()
    fig_reranking()
    fig_deknot()
    print("Done.")
