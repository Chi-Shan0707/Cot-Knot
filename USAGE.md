# Usage Guide for code-not-text

This guide provides detailed instructions for reproducing the experiments and figures in the workshop paper "Cross-Domain Limits of Hand-Crafted CoT-Surface Features".

## Prerequisites

### Required Data

Most scripts require access to experiment data that is not included in this repository due to size constraints. The scripts expect data at specific paths:

- Math experiments: `/path/to/math_experiments/`
- Science experiments: `/path/to/science_experiments/`
- Coding experiments: `/path/to/coding_experiments/`
- GLM annotations: `/path/to/glm_annotations/`

### Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

For GLM-based knot annotation:
```bash
pip install zhipuai
```

## Figure Generation

### Generate All Paper Figures

```bash
cd workshop/cotknot
python scripts/gen_figures_v12_5.py
```

This will generate:
- `figures/fig_auroc_by_anchor_v12_5.pdf` - AUROC curves and AoA bar chart
- `figures/fig_ssl_ceiling_v12_5.pdf` - SSL ceiling comparison
- `figures/fig_reranking_v12_5.pdf` - Best-of-N reranking results
- `figures/fig_deknot_alldomains_v12_5.pdf` - De-knotting ablation results

### Individual Figure Scripts

If you have the required data locally, you can modify the paths in `gen_figures_v12_5.py` and run:
```bash
python scripts/gen_figures_v12_5.py
```

## De-knotting Experiments

### V2: All Domains with Proper Token-Level Masking

```bash
# Requires: GLM API key, NAD cache
cd /path/to/NAD_Next
source .venv/bin/activate
python /path/to/workshop/cotknot/scripts/deknot_alldomains_v2.py
```

Output: `results/tables/deknot_alldomains_v2.csv`

### V1: Coding-Only De-knotting

```bash
python scripts/deknot_coding_experiment.py
```

## GLM Knot Annotation

### Math Knots

```bash
# Run GLM labeling
python scripts/run_glm_math_knot_labeling_v4.py

# Analyze results
python scripts/analyze_glm_math_knot.py
```

### Science Knots

```bash
# Run GLM labeling (v3)
python scripts/run_glm_science_knot_labeling_v3.py

# Analyze results
python scripts/analyze_glm_science_knot_v2.py
```

### Coding Knots

```bash
# Run GLM labeling
python scripts/run_glm_coding_knot_labeling_v4.py

# Analyze cross-domain patterns
python scripts/analyze_glm_knot_v4.py
python scripts/compare_glm_knot_v4_domains.py
```

## Cross-Domain Analysis

### Compare Knot Patterns Across Domains

```bash
python scripts/compare_glm_knot_v4_domains.py
```

### Build Knot Error Enrichment Analysis

```bash
python scripts/build_glm_knot_error_enrichment_v4.py
```

## Paper Compilation

### Compile Latest Version (v13)

```bash
cd paper
pdflatex paper_v13.tex
pdflatex paper_v13.tex  # Run twice for references
```

### Clean Build Artifacts

```bash
cd paper
rm -f *.aux *.log *.out *.fls *.fdb_latexmk *.run.xml *.synctex.gz
```

## Result Tables

Key result tables are located in `results/tables/`:

- `aoa_bootstrap_ci.csv` - AoA bootstrap confidence intervals
- `bon_reranking_domain_pass1_ci.csv` - Best-of-N reranking results
- `deknot_alldomains_v2.csv` - De-knotting ablation results
- `glm_knot_findings_v4.md` - Knot annotation summary

## Common Issues

### Path Errors

Most scripts use hardcoded paths. You'll need to update these to match your local setup:

```python
# Example: Update data paths
DATA_BASE = "/your/path/to/experiments"
```

### Missing Dependencies

If you encounter import errors:
```bash
pip install -r requirements.txt  # If available
# Or install individually
pip install numpy pandas matplotlib seaborn scikit-learn
```

### GLM API Issues

For knot annotation scripts requiring GLM:
1. Set your API key: `export ZHIPUAI_API_KEY="your-key-here"`
2. Ensure you have sufficient API quota
3. Check network connectivity

## Citation

If you use this code or find the research helpful, please cite:

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

For questions or issues, please contact: yhchi25@m.fudan.edu.cn
