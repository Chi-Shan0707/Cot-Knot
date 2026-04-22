<div align="center">

# Cot-Knot

### CoT Quality Signals Are Domain-Conditioned: A Measurement Perspective on Different Knot Types in Math and Coding

<p>
  <img src="https://img.shields.io/badge/status-workshop%20paper-blue" alt="status" />
  <img src="https://img.shields.io/badge/focus-measurement%20perspective-6f42c1" alt="focus" />
  <img src="https://img.shields.io/badge/theme-math%20vs%20coding-0a7e5a" alt="theme" />
  <img src="https://img.shields.io/badge/license-Apache--2.0-orange" alt="license" />
</p>

*Self-correction is not a universal quality signal.*

[Paper Overview](#paper-overview) ·
[Main Claims](#main-claims) ·
[Knot Taxonomy](#knot-taxonomy) ·
[Annotation Validity](#annotation-validity) ·
[Repository Structure](#repository-structure) ·
[Citation](#citation)

</div>

`Cot-Knot` is a paper repository centered on one claim:

> CoT quality signals — including compact verifier probes built on self-correction features — are **domain-conditioned** rather than universal.

The project takes a **measurement perspective**: it asks not only whether a probe predicts correctness, but whether the annotation protocol that grounds the probe is itself valid across domains.

---

## Paper Overview

The paper studies when a compact probe built on chain-of-thought self-correction features carries reliable information about answer quality. The central finding is that the same surface pattern of "revision" or "self-correction" does not behave identically across domains — and that the annotation protocol needed to operationalize the concept breaks down differently in math vs. coding.

The measurement framing distinguishes this project from standard verifier benchmarking. Rather than asking only "does the probe score correlate with correctness?", it asks: "is the construct being measured the same construct across domains?"

The paper's current answer is **no**. Math knots are explicit, text-visible local state breaks that annotators agree on and that predict correctness. Coding knots, as originally formulated, are annotation-invalid and predictively weak. The project proposes a replacement coding protocol (execution-semantic breaks) that improves both annotation agreement and conceptual precision.

### Research question

> **When does self-correction in chain-of-thought behave like a stable quality signal, and when does it stop meaning the same thing?**

The treatment is a measurement and interpretation problem, not a pure modeling problem.

## Main claims

The paper makes three linked empirical claims:

1. **Compact CoT-based quality probes are strong in math, reasonable in science, and weak in coding.** (AUC-of-AUROC: math 0.958, science 0.799, coding 0.434 — below the 0.506 token-confidence baseline.)
2. **Math and coding exhibit different knot types rather than one universal self-correction phenomenon.** The domain difference is not just quantitative but reflects a construct-validity failure for coding under the original annotation protocol.
3. **RL posttraining can preserve ranking signal while breaking the original binary operating point of an SFT-era verifier.** (FPR rises from 0.982 → 0.998; only 8 / 4352 wrong runs rejected at step 1000.)

### High-level picture

| Domain | Probe AUROC@100% | AUC-of-AUROC | Annotation validity |
|---|---|---|---|
| **Math** | 0.982 | 0.958 | κ = 0.603 (acceptable) |
| **Science** | 0.841 | 0.799 | — |
| **Coding** | < 0.5 | 0.434 | κ = 0.0 (original); κ = 0.329 (execution-break protocol) |

## Knot taxonomy

In this repository, a **knot** is a mechanism-level local entanglement or break in reasoning state.

Typical examples include:

- explicit contradiction
- incompatible redefinition
- leaked case frame
- unresolved repair
- lost execution state
- patchy backtracking
- branch, loop, or index-state inconsistency

The important claim is not merely that knots occur, but that **their semantics depend on domain**.

### Working interpretation

- **Math**: a knot can be explicit, local, and still compatible with successful recovery. Annotators agree reliably.
- **Coding**: a similar-looking correction can mask a deeper execution-state failure. The original annotation protocol is invalid (κ = 0.0). A replacement execution-break protocol improves agreement (κ = 0.329) but remains below the threshold for strong claims.
- **Takeaway**: the same textual behavior can correspond to different underlying mechanisms, and different annotation protocols are needed in each domain.

### Methodological stance

This repository takes a deliberately conservative stance on interpretation:

- it separates **full-scale quantitative signal** from **pilot mechanism evidence**
- it treats annotation-derived knot labels as mechanism probes rather than universal ground truth
- it emphasizes **domain-conditioned interpretation** over universal verifier narratives
- it treats RL warning results primarily as evidence of **operating-point failure under regime shift**
- it explicitly reports annotation validity (κ) as a first-class result, not a footnote

## Annotation validity

**This is a critical disclosure.**

The knot annotation results are reported with their inter-annotator agreement statistics:

| Protocol | Domain | n | κ | Validity |
|---|---|---|---|---|
| GLM knot labels (original) | Math | 30 | 0.603 | Acceptable |
| GLM knot labels (original) | Coding | 30 | 0.000 | **INVALID** — GLM marks 100% positive, humans 6.7% |
| Execution-break protocol | Coding | 56 | 0.329 | Below threshold |

The original coding annotation protocol is invalid and all results derived from it are not reported as primary findings. The execution-break protocol is the replacement, but its agreement (κ = 0.329) is still below the κ ≥ 0.6 threshold for strong claims. Coding knot mechanism results are therefore treated as pilot-scale evidence only.

## Repository structure

```
cotknot/
├── paper/
│   ├── paper_v8.tex          # manuscript source (v8, measurement framing)
│   └── paper_v8.pdf          # compiled PDF
├── figures/
│   └── fig_knot_domain_profiles_v3.pdf   # main paper figure
├── scripts/
│   ├── run_glm_math_knot_labeling.py     # GLM annotation pipeline (math)
│   ├── run_glm_math_knot_spans.py        # span extraction
│   ├── verify_glm_math_knot_spans.py     # span verification
│   ├── analyze_glm_math_knot.py          # knot label analysis
│   ├── analyze_glm_math_knot_spans.py    # span-level analysis
│   ├── compare_knot_domain_features.py   # cross-domain feature comparison → temporal profiles CSV
│   └── plot_knot_domain_profiles.py      # figure generation (reads temporal profiles CSV)
└── README.md
```

The figure pipeline is:

```
run_glm_math_knot_labeling.py
        ↓
compare_knot_domain_features.py  →  knot_cross_domain_temporal_profiles_pilot64.csv
        ↓
plot_knot_domain_profiles.py  →  fig_knot_domain_profiles_v3.pdf
```

## Status

This repository holds the paper-facing materials for a workshop submission (target: NeurIPS Math-AI / ICLR SCSL / EMNLP BlackboxNLP workshop tier).

The current version (v8) implements the measurement framing. The paper and main figure are committed. Scripts supporting the annotation and comparison pipeline are included under `scripts/`.

Known limitations (to be addressed in revision):

- Coding annotation validity is below threshold; stronger IAA required for main-venue claims.
- No competitive baseline comparison (Math-Shepherd, ProcessBench) yet included.
- RL warning section requires additional novelty framing.

## License

This repository is released under the Apache 2.0 License. See `LICENSE`.

## Citation

```bibtex
@misc{chi2026cotknot,
  title        = {CoT Quality Signals Are Domain-Conditioned: A Measurement Perspective on Different Knot Types in Math and Coding},
  author       = {Yuhan Chi},
  year         = {2026},
  note         = {Workshop paper repository}
}
```

## Contact

Yuhan Chi
Fudan University
