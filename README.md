<div align="center">

# code-not-text

### Cross-Domain Limits of Hand-Crafted CoT-Surface Features:<br>Strong in Math, Narrow in Science, Weak in Coding

<p>
  <img src="https://img.shields.io/badge/status-workshop%20paper%20v13-blue" alt="status" />
  <img src="https://img.shields.io/badge/methods-5%20convergent-6f42c1" alt="methods" />
  <img src="https://img.shields.io/badge/domains-math%20%7C%20science%20%7C%20coding-0a7e5a" alt="domains" />
  <img src="https://img.shields.io/badge/license-Apache--2.0-orange" alt="license" />
</p>

*Hand-crafted CoT-surface features transfer well to math, only partly to science, and poorly to coding correctness on unseen problems.*

[Core Finding](#core-finding) ·
[Five Methods](#five-methods-one-ceiling) ·
[Results](#results) ·
[Repository Structure](#repository-structure) ·
[Reproduce](#reproduce) ·
[Citation](#citation)

</div>

---

## Core Finding

Can cheap summaries of chain-of-thought (CoT) traces predict whether a solution is correct? We test a single feature family—hand-crafted CoT-surface and token-trajectory features—on **DeepSeek-R1-0528-Qwen3-8B** across three domains: math, science, and coding.

**It depends on the domain.** In math, the signal is strong and emerges early. In science, the signal is usable but narrower and largely confidence-driven. In coding, the signal is weak and does not transport cleanly to unseen problems.

| Domain | AoA (AUC-of-AUROC) | AUROC@100% | Reranking Δ pass@1 |
|:--------|:------------------:|:----------:|:------------------:|
| **Math** | **0.958** [0.931, 0.980] | **0.982** | **+10.0%** |
| **Science** | **0.799** [0.775, 0.822] | **0.841** | **+8.0%** |
| **Coding** | 0.434 [0.404, 0.464] | 0.407 | −0.6% |

**Key insight**: These features measure different underlying constructs across domains. In math, they capture "failure to converge" in reasoning. In coding, where correctness requires execution, the same features measure exploratory effort—a mismatch between the measurement instrument and the target variable.

---

## Five Methods, One Ceiling

| # | Method | Best coding result | Verdict |
|---|--------|-------------------|---------|
| 1 | Interpretable probe (LogReg + SVD) | AoA 0.434 / AUROC@100 0.407 | Below baseline |
| 2 | Broad feature sweep (83 scalars) | AUROC 0.556 (top-4 combo) | Weak only |
| 3 | Nonlinear MLP classifier | AUROC 0.499–0.507 | No meaningful gain |
| 4 | SSL pre-training (42K unlabeled runs) | 0.537 at lf=5%; **0.454 at lf=100%** | Low-label regularizer only |
| 5 | Token-level de-knotting ablation | AUROC Δ = +0.006 | Neutral |

**Why the convergence matters.** Five independent approaches—interpretable probes, feature engineering, nonlinear models, self-supervised learning, and surface-noise removal—all reach the same ceiling for coding. This rules out several plausible failure modes:

- **Not model capacity**: MLP classifiers don't escape the ceiling
- **Not label scarcity**: SSL on 42K unlabeled traces still fails at full labels
- **Not surface noise**: De-knotting doesn't recover signal
- **Not feature engineering**: 83 coding-specific scalars don't close the gap

The evidence supports a measurement explanation: code correctness depends on executable state evolution, which is not captured by token-trajectory summaries.

---

## Results

All quantitative claims reproduce from `results/tables/`. Key tables:

| File | Contents |
|------|----------|
| `results/tables/bon_reranking_domain_pass1_ci.csv` | Best-of-N=64 pass@1 with bootstrap 95% CI |
| `results/tables/deknot_alldomains_v2.csv` | De-knotting ablation: all 3 domains |
| `results/tables/glm_knot_findings_v4.md` | Knot annotation summary (math/science/coding) |
| `results/tables/aoa_bootstrap_ci.csv` | AoA bootstrap CI by domain |
| `results/tables/coding_feature_family_ablation.csv` | Grouped feature ablation results |
| `results/tables/cot_run_judge_scores.csv` | CoT-only run judge OOF predictions |

### Cross-domain feature importance

Permutation importance reveals different semantics across domains:

- **Math**: `traj_reflection_count` (0.242), `traj_continuity` (0.092), `traj_novelty` (0.052) — trajectory-shaped signal
- **Science**: `tok_gini_tail` (0.038), `tok_conf_recency` (0.033), `tok_conf_prefix` (0.033) — confidence-heavy signal
- **Coding**: No strong, stable feature importance pattern

### SSL results

Self-supervised pre-training shows domain-specific effects:

| Domain | no_svd_lr (lf=5%) | ssl_v2_r24 (lf=5%) | Δ | ssl_v2_r24 (lf=100%) |
|--------|:-----------------:|:------------------:|:-:|:--------------------:|
| Math    | 0.898 | 0.896 | −0.2pp | 0.944 |
| Science | 0.607 | **0.674** | +6.7pp | 0.762 |
| Coding  | 0.488 | **0.537** | +4.9pp | **0.454** ← ceiling persists |

SSL helps in low-label regimes but doesn't change the full-label ceiling.

---

## Repository Structure

```
code-not-text/
├── paper/
│   ├── paper_v13.tex           # Current paper source (v13, cross-domain focus)
│   └── paper_v13.pdf           # Compiled PDF
├── figures/
│   ├── fig_auroc_by_anchor_v12_5.pdf   # Fig 1: AUROC curves + AoA bar chart
│   ├── fig_ssl_ceiling_v12_5.pdf       # Fig 2: SSL ceiling — coding vs science
│   ├── fig_reranking_v12_5.pdf         # Fig 3: Best-of-N reranking pass@1
│   ├── fig_deknot_alldomains_v12_5.pdf # Fig 4: De-knotting ablation (all 3 domains)
│   └── fig_knot_domain_profiles_v3.pdf  # Legacy: feature gap in break-positive subsets
├── scripts/
│   ├── gen_figures_v12_5.py            # Generate all 4 paper figures
│   ├── deknot_alldomains_v2.py         # De-knotting experiment v2 (all domains)
│   ├── deknot_coding_experiment.py     # De-knotting experiment v1 (coding only)
│   ├── analyze_glm_knot_v4.py          # Cross-domain knot analysis
│   ├── compare_glm_knot_v4_domains.py  # Domain comparison utilities
│   ├── run_glm_*_knot_labeling*.py     # GLM knot annotation pipelines
│   └── knot_glm_common.py              # Shared utilities
├── results/
│   └── tables/                          # All CSV/MD result tables
├── temp/                               # Old drafts, review notes (gitignored)
├── .gitignore
├── LICENSE                             # Apache 2.0
└── README.md
```

---

## Reproduce

### Figures (requires local data paths)

```bash
cd workshop/cotknot
python scripts/gen_figures_v12_5.py
# Outputs: figures/fig_*_v12_5.pdf
```

### De-knotting experiment (requires GLM API key + NAD cache)

```bash
# v2: all domains, proper token-level masking
cd /path/to/NAD_Next && source .venv/bin/activate
python /path/to/scripts/deknot_alldomains_v2.py
# Results: results/tables/deknot_alldomains_v2.csv
```

### Compile paper

```bash
cd paper
pdflatex paper_v13.tex && pdflatex paper_v13.tex
```

---

## Paper Summary

**Model:** DeepSeek-R1-0528-Qwen3-8B
**Data:** AIME24/25 + BRUMO25 + HMMT25 (math), GPQA (science), LiveCodeBench-v5 (coding)
**Features:** `tok_conf`, `tok_gini`, `tok_logprob`, `tok_neg_entropy`, `tok_selfcert` (per-token) +
`traj_continuity`, `traj_novelty`, `traj_reflection_count` (trajectory) + `nc_mean` (neuron)

**Central claim**: Within this single-model study, cheap hand-crafted CoT-surface features are domain-specific measurement tools. They work very well in math, partly in science, and poorly in coding.

**What this does NOT claim**: This does not rule out raw-text classifiers, code-aware models, execution-aware selectors, hidden-state probes, or LLM judges. A more likely explanation is that coding needs different signals, such as execution-aware ones.

---

## License

Apache 2.0. See `LICENSE`.

## Citation

```bibtex
@misc{chi2026codenottext,
  title   = {Cross-Domain Limits of Hand-Crafted CoT-Surface Features:
             Strong in Math, Narrow in Science, Weak in Coding},
  author  = {Yuhan Chi},
  year    = {2026},
  note    = {Workshop paper. Code: https://github.com/Chi-Shan0707/code-not-text}
}
```

## Contact

Yuhan Chi · Fudan University · `yhchi25@m.fudan.edu.cn`
