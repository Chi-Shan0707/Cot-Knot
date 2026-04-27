<div align="center">

# cot-signal

### Code Correctness Is Not in the Text:<br>Five Methods, One Ceiling — Convergent Evidence for CoT Quality Measurement Breakdown in Coding

<p>
  <img src="https://img.shields.io/badge/status-workshop%20paper%20v12-blue" alt="status" />
  <img src="https://img.shields.io/badge/methods-5%20convergent-6f42c1" alt="methods" />
  <img src="https://img.shields.io/badge/domains-math%20%7C%20science%20%7C%20coding-0a7e5a" alt="domains" />
  <img src="https://img.shields.io/badge/license-Apache--2.0-orange" alt="license" />
</p>

> **Suggested new name for this repository:** `cot-signal`
> *(The current name `Cot-Knot` was coined early in the project around the knot-annotation framing.
> The paper has grown into a broader measurement study of whether CoT traces carry quality signals at all —
> `cot-signal` is shorter, more general, and maps directly to the central question.)*

*Within this feature family, code correctness is not in the text.*

[Core Finding](#core-finding) ·
[Five Methods](#five-methods-one-ceiling) ·
[Results](#results) ·
[Repository Structure](#repository-structure) ·
[Reproduce](#reproduce) ·
[Citation](#citation)

</div>

---

## Core Finding

We test whether token-trajectory features (per-token entropy signals + text-trajectory statistics)
encode coding correctness in chain-of-thought traces.
We use **five independent methods** on the same feature family, applied to three domains
(math competitions, science, coding) with a single model — **DeepSeek-R1-0528-Qwen3-8B**.

**All five methods converge on the same answer for coding: no.**

| Domain | AUC-of-AUROC | AUROC@100% | Reranking Δ pass@1 |
|--------|:------------:|:----------:|:------------------:|
| **Math** | **0.958** [0.931, 0.980] | **0.982** | **+10.0%** |
| **Science** | **0.799** [0.775, 0.822] | **0.841** | **+8.0%** |
| Coding | 0.434 | 0.407 | −0.6% |

The coding probe falls *below* a single-feature token-confidence baseline (0.506). No method
in our study lifts coding AUROC meaningfully above chance at full labels.

---

## Five Methods, One Ceiling

| # | Method | Best coding result | Verdict |
|---|--------|-------------------|---------|
| 1 | Interpretable probe (LogReg + SVD) | AoA 0.434 / AUROC@100 0.407 | Below baseline |
| 2 | Broad feature sweep (83 scalars) | AUROC 0.556 (top-4 combo) | Marginal |
| 3 | Nonlinear MLP classifier | AUROC 0.499–0.507 | No change |
| 4 | **SSL pre-training** (42K unlabeled runs) | 0.537 at lf=5%; **0.454 at lf=100%** | Reg. only; ceiling persists |
| 5 | Token-level de-knotting ablation | AUROC Δ = +0.006 | Neutral |

**Why SSL is the decisive experiment.** Self-supervised pre-training on 42,000 unlabeled CoT
traces — including a domain-specific variant with VICReg anti-collapse, 5-term structured loss,
and 500 training epochs — tops out near AUROC 0.5 for coding at full labels. The training loss
decreases (structure is learned), but AUROC stays flat (correctness is not part of that structure).
This rules out label scarcity as the bottleneck.

**Why the de-knotting result matters.** Properly removing "knot" token spans (circular reasoning segments)
from all five per-token signal arrays via fast-tokenizer offset mapping:
- Math: AUROC *drops* by 0.049 — knot tokens are the discriminative signal itself
- Coding: AUROC changes by +0.006 — no masking strategy recovers signal

---

## Results

All quantitative claims reproduce from `results/tables/`. Key tables:

| File | Contents |
|------|----------|
| `results/tables/bon_reranking_domain_pass1_ci.csv` | Best-of-N=64 pass@1 with bootstrap 95% CI |
| `results/tables/deknot_alldomains_v2.csv` | De-knotting ablation: all 3 domains, v2 (proper token masking) |
| `results/tables/glm_knot_findings_v4.md` | Knot annotation summary (math/science/coding) |
| `results/tables/aoa_bootstrap_ci.csv` | AoA bootstrap CI by domain |

### SSL results

SSL experiments are documented in `scripts/` and full results in
`/home/jovyan/work/SVDomain/results/tables/domain_specific_ssl_v2.csv` (not in this repo — path-local).

Key SSL v2 numbers at anchor=100%:

| Domain | no_svd_lr (lf=5%) | ssl_v2_r24 (lf=5%) | Δ | ssl_v2_r24 (lf=100%) |
|--------|:-----------------:|:------------------:|:-:|:--------------------:|
| Math    | 0.898 | 0.896 | −0.2pp | 0.944 |
| Science | 0.607 | **0.674** | +6.7pp | 0.762 |
| Coding  | 0.488 | **0.537** | +4.9pp | **0.454** ← ceiling |

---

## Repository Structure

```
cot-signal/             ← (suggested rename from Cot-Knot)
├── paper/
│   ├── paper_v12.tex           # Current paper source (v12, multi-method)
│   └── paper_v12.pdf           # Compiled PDF
├── figures/
│   ├── fig_auroc_by_anchor_v12.pdf   # Fig 1: AUROC curves + AoA bar chart
│   ├── fig_ssl_ceiling_v12.pdf       # Fig 2: SSL ceiling — coding vs science
│   ├── fig_reranking_v12.pdf         # Fig 3: Best-of-N reranking pass@1
│   ├── fig_deknot_alldomains_v12.pdf # Fig 4: De-knotting ablation (all 3 domains)
│   └── fig_knot_domain_profiles_v3.pdf  # Legacy: feature gap in break-positive subsets
├── scripts/
│   ├── gen_figures_v12.py            # Generate all 4 paper figures (requires local data paths)
│   ├── deknot_alldomains_v2.py       # De-knotting experiment v2 (all domains, token masking)
│   ├── deknot_coding_experiment.py   # De-knotting experiment v1 (coding only)
│   ├── plot_knot_domain_profiles.py  # Legacy figure script
│   ├── run_glm_*_knot_labeling*.py   # GLM knot annotation pipelines
│   ├── analyze_glm_knot_v4.py        # Cross-domain knot analysis
│   ├── compare_glm_knot_v4_domains.py
│   └── build_glm_knot_error_enrichment_v4.py
├── results/
│   └── tables/                       # All CSV/MD result tables
├── temp/                             # Old drafts, review notes (gitignored)
├── .gitignore
├── LICENSE
└── README.md
```

---

## Reproduce

### Figures (requires local data paths)

```bash
cd workshop/cotknot
python scripts/gen_figures_v12.py
# Outputs: figures/fig_*_v12.pdf
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
pdflatex paper_v12.tex && pdflatex paper_v12.tex
```

---

## Paper Summary

**Model:** DeepSeek-R1-0528-Qwen3-8B
**Data:** AIME24/25 + BRUMO25 + HMMT25 (math), GPQA (science), LiveCodeBench-v5 (coding)
**Features:** `tok_conf`, `tok_gini`, `tok_logprob`, `tok_neg_entropy`, `tok_selfcert` (per-token) +
`traj_continuity`, `traj_novelty`, `traj_reflection_count` (trajectory)

The central claim: these features measure "failure to converge" in math (where convergence has a
closed-form answer) but measure "exploratory effort" in coding (where correctness requires execution).
Same feature name, different underlying construct — a measurement non-invariance.

---

## License

Apache 2.0. See `LICENSE`.

## Citation

```bibtex
@misc{chi2026cotsignal,
  title   = {Code Correctness Is Not in the Text: Five Methods, One Ceiling ---
             Convergent Evidence for {CoT} Quality Measurement Breakdown in Coding},
  author  = {Yuhan Chi},
  year    = {2026},
  note    = {Workshop paper. Code: https://github.com/Chi-Shan0707/Cot-Knot}
}
```

## Contact

Yuhan Chi · Fudan University · `masterwuguicyh@gmail.com`
