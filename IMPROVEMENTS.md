# cotknot Project GitHub Submission Summary

## Overview

This document summarizes the improvements made to the `cotknot` project for GitHub submission, based on the v12 and v13 workshop papers.

## Repository Name & Purpose

**Current Name**: `code-not-text`

**Rationale**: The name directly captures the core finding—code correctness cannot be reliably determined from text-based CoT surface features alone. This is concise, memorable, and accurately reflects the research contribution.

**Alternative Considered**: `cot-domain-limits` was considered but rejected as less memorable and more technical.

## Key Improvements Made

### 1. Documentation Overhaul

**README.md Updates**:
- Updated to reflect v13 cross-domain findings
- Restructured for clarity with better flow
- Added explicit domain-specific results table
- Clarified what the paper does and does NOT claim
- Improved citation format

**New Documentation Files**:
- `CONTRIBUTING.md`: Guidelines for contributors
- `USAGE.md`: Detailed usage instructions for all scripts
- `IMPROVEMENTS.md`: This file documenting changes

### 2. Version Control & Git Management

**.gitignore Optimization**:
- Updated to track only paper_v13.tex/.pdf as latest versions
- Ignore intermediate versions (v7.5-v12.5) to reduce clutter
- Ignore intermediate LaTeX build artifacts
- Ignore intermediate figure versions, keeping only v12_5 as latest
- Ignore intermediate script versions

**Commit History**:
- Created clear, descriptive commits for each change
- Maintained linear history with atomic changes
- All commits signed off with meaningful messages

### 3. File Organization

**Paper Structure**:
- Primary: `paper_v13.tex/.pdf` (latest cross-domain version)
- Archived: Earlier versions kept for reference but gitignored

**Figures Structure**:
- Current: `fig_*_v12_5.pdf` (latest versions used in v13)
- Legacy: `fig_knot_domain_profiles_v3.pdf` (kept for historical context)

**Scripts Structure**:
- Current: `gen_figures_v12_5.py` (latest figure generation)
- Analysis scripts: Cross-domain knot analysis, GLM annotation pipelines
- All scripts have clear docstrings and usage comments

### 4. Dependencies & Reproducibility

**requirements.txt**:
- Added explicit dependency specification
- Listed core packages with version requirements
- Commented optional development dependencies
- Made setup process clearer for new users

### 5. License & Legal

**License**: Apache 2.0 (already present, confirmed appropriate)

**Citation Format**:
```bibtex
@misc{chi2026codenottext,
  title   = {Cross-Domain Limits of Hand-Crafted CoT-Surface Features:
             Strong in Math, Narrow in Science, Weak in Coding},
  author  = {Yuhan Chi},
  year    = {2026},
  note    = {Workshop paper. Code: https://github.com/Chi-Shan0707/code-not-text}
}
```

## What to Submit to GitHub

### Essential Files
- `README.md` - Updated with v13 findings
- `LICENSE` - Apache 2.0
- `requirements.txt` - Dependencies
- `CONTRIBUTING.md` - Contribution guidelines
- `USAGE.md` - Detailed usage instructions
- `paper/paper_v13.tex` - Latest paper source
- `paper/paper_v13.pdf` - Latest paper PDF
- `.gitignore` - Optimized for version tracking
- `scripts/gen_figures_v12_5.py` - Figure generation
- `scripts/analyze_glm_knot_v4.py` - Cross-domain analysis
- `figures/fig_*_v12_5.pdf` - All four main figures
- `results/tables/*.csv` - Key result tables

### Optional but Recommended
- Additional analysis scripts for reproducibility
- Legacy figure `fig_knot_domain_profiles_v3.pdf`
- Detailed knot annotation scripts

### What NOT to Submit
- `temp/` folder (gitignored)
- Intermediate paper versions
- LaTeX build artifacts
- Personal API keys or local paths
- Very large raw data files

## Repository Structure for Submission

```
code-not-text/
├── README.md (updated)
├── LICENSE
├── requirements.txt (new)
├── CONTRIBUTING.md (new)
├── USAGE.md (new)
├── .gitignore (updated)
├── paper/
│   ├── paper_v13.tex (new)
│   └── paper_v13.pdf (new)
├── figures/
│   ├── fig_auroc_by_anchor_v12_5.pdf
│   ├── fig_ssl_ceiling_v12_5.pdf
│   ├── fig_reranking_v12_5.pdf
│   ├── fig_deknot_alldomains_v12_5.pdf
│   └── fig_knot_domain_profiles_v3.pdf
├── scripts/
│   ├── gen_figures_v12_5.py
│   ├── analyze_glm_knot_v4.py
│   ├── compare_glm_knot_v4_domains.py
│   └── [other analysis scripts]
└── results/
    └── tables/
        ├── aoa_bootstrap_ci.csv
        ├── bon_reranking_domain_pass1_ci.csv
        ├── deknot_alldomains_v2.csv
        └── [key result tables]
```

## GitHub Repository Setup

### Repository Name
**Recommendation**: Keep `code-not-text`

**Description**:
```
Cross-domain limits of hand-crafted CoT-surface features: Research showing these features work well in math, partly in science, and poorly in coding correctness prediction.
```

### Topics/Tags
- chain-of-thought
- code-generation
- verification
- cross-domain-analysis
- nlp-research
- workshop-paper

### About Section
```
This repository contains code and artifacts for the workshop paper "Cross-Domain Limits of Hand-Crafted CoT-Surface Features: Strong in Math, Narrow in Science, Weak in Coding."

Key Finding: Hand-crafted CoT-surface features transfer well to math, only partly to science, and poorly to coding correctness on unseen problems.

The study uses DeepSeek-R1-0528-Qwen3-8B across three domains: math competitions (AIME/BRUMO/HMMT), science (GPQA), and coding (LiveCodeBench-v5), using five convergent methods to demonstrate domain-specific measurement limits.
```

## Next Steps for User

1. **Review all changes**: Check that updates accurately reflect the research
2. **Test repository locally**: Ensure all paths and scripts work as documented
3. **Create GitHub repository**: Initialize with the recommended settings above
4. **Push cleaned commits**: The 6 commits made during this session are ready to push
5. **Add additional documentation**: Consider adding CHANGELOG.md if planning continued development

## Commits Made During This Session

1. `Update .gitignore: track paper_v13.pdf as latest version, ignore intermediate LaTeX files`
2. `Update README: reflect v13 cross-domain findings, restructure for clarity`
3. `Add paper v13: cross-domain limits of hand-crafted CoT-surface features`
4. `Update .gitignore: ignore intermediate paper versions, figures, and scripts; keep only v13 and v12_5 as latest`
5. `Add CONTRIBUTING.md and USAGE.md for better repository documentation`
6. `Add requirements.txt with core project dependencies`

All commits are atomic, well-described, and ready for public repository submission.

## Contact & Attribution

**Author**: Yuhan Chi
**Institution**: Fudan University
**Email**: yhchi25@m.fudan.edu.cn
**License**: Apache 2.0

The repository is structured for maximum reproducibility and minimal maintenance burden while preserving all essential research artifacts.
