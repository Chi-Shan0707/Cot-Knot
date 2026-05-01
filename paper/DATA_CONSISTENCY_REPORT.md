# Data Consistency and Authenticity Verification Report
## Paper: paper_v13.tex

**Date:** 2026-05-01
**Analyzer:** Comprehensive Academic Quality Check

---

## Executive Summary

This report provides a comprehensive verification of data consistency between the paper (paper_v13.tex), README.md, and actual data files in the repository.

### Critical Findings

✅ **CONSISTENT:** Core numerical claims match between paper and README
⚠️ **MISSING DATA:** Several data files referenced in paper/README do not exist
✅ **VERIFIED:** De-knotting experimental data exists and matches claims

---

## 1. Core Numerical Claims Verification

### 1.1 AUROC Values

| Domain | Paper Claim | README Claim | Status |
|:--------|:-----------|:-------------|:-------|
| Math | 0.958 [0.931, 0.980] | 0.958 [0.931, 0.980] | ✅ MATCH |
| Science | 0.799 [0.775, 0.822] | 0.799 [0.775, 0.822] | ✅ MATCH |
| Coding | 0.434 [0.404, 0.464] | 0.434 [0.404, 0.464] | ✅ MATCH |

### 1.2 Best-of-N Reranking Results

| Domain | Paper Claim | README Claim | Status |
|:--------|:-----------|:-------------|:-------|
| Math | +10.0% | +10.0% | ✅ MATCH |
| Science | +8.0% | +8.0% | ✅ MATCH |
| Coding | -0.6% | -0.6% | ✅ MATCH |

### 1.3 Sample Sizes

| Domain | Paper Claim | README Claim | Status |
|:--------|:-----------|:-------------|:-------|
| Math | 7,680 runs | Not explicitly stated | ⚠️ PARTIAL |
| Science | 12,672 runs | Not explicitly stated | ⚠️ PARTIAL |
| Coding | 10,688 runs | Not explicitly stated | ⚠️ PARTIAL |

**Note:** Sample sizes are claimed in paper but not explicitly broken down in README summary table.

### 1.4 Model Specification

| Attribute | Paper Claim | README Claim | Status |
|:----------|:-----------|:-------------|:-------|
| Model Name | DeepSeek-R1-0528-Qwen3-8B | DeepSeek-R1-0528-Qwen3-8B | ✅ MATCH |
| Temperature | 1.0 | Not specified | ⚠️ PARTIAL |

---

## 2. Critical Data Files Status

### 2.1 Files Referenced in Paper/README

| File | Referenced In | Status | Notes |
|:-----|:--------------|:-------|:------|
| `results/tables/bon_reranking_domain_pass1_ci.csv` | README | ❌ NOT FOUND | Critical for reranking claims |
| `results/tables/aoa_bootstrap_ci.csv` | README | ❌ NOT FOUND | Critical for AUROC confidence intervals |
| `results/tables/coding_feature_family_ablation.csv` | README | ❌ NOT FOUND | Critical for coding ablation results |
| `results/tables/cot_run_judge_scores.csv` | README | ❌ NOT FOUND | Critical for CoT-only judge results |
| `results/tables/deknot_alldomains_v2.csv` | Paper, README | ✅ FOUND | Verified content |

### 2.2 Verified Data: De-knotting Experiment

**File:** `/home/jovyan/work/SVDomain/workshop/cotknot/results/tables/deknot_alldomains_v2.csv`

```csv
domain,n_runs,n_knotted,knot_rate_correct,knot_rate_incorrect,total_chars_removed,auroc_original,auroc_deknot,delta,verdict
math,200,155,0.73,0.82,153270,0.5018,0.4528,-0.0489,HURTS
science,200,113,0.6,0.53,173523,0.5269,0.5214,-0.0055,NEUTRAL (no change)
coding,200,183,0.93,0.9,138273,0.4602,0.4658,0.0056,NEUTRAL (no change)
```

**Verification Against Paper Claims:**

| Domain | Paper AUROC Δ | File AUROC Δ | Status |
|:--------|:-------------|:-------------|:-------|
| Math | -0.049 | -0.0489 | ✅ MATCH |
| Science | -0.006 | -0.0055 | ✅ MATCH |
| Coding | +0.006 | +0.0056 | ✅ MATCH |

**Analysis:** The de-knotting data is consistent and verified.

---

## 3. Missing Data Impact Analysis

### 3.1 High-Impact Missing Files

1. **`bon_reranking_domain_pass1_ci.csv`**
   - **Claim Impact:** Best-of-N=64 reranking results (+10.0%, +8.0%, -0.6%)
   - **Risk Level:** HIGH
   - **Recommendation:** This file must exist or claims must be softened

2. **`aoa_bootstrap_ci.csv`**
   - **Claim Impact:** All AUROC confidence intervals [0.931, 0.980], etc.
   - **Risk Level:** HIGH
   - **Recommendation:** Critical for statistical validity claims

3. **`coding_feature_family_ablation.csv`**
   - **Claim Impact:** Grouped feature ablation results (traj_only: 0.509, etc.)
   - **Risk Level:** MEDIUM
   - **Recommendation:** Important for coding robustness claims

4. **`cot_run_judge_scores.csv`**
   - **Claim Impact:** CoT-only judge OOF AUC 0.481 ± 0.053
   - **Risk Level:** MEDIUM
   - **Recommendation:** Important for judge model claims

### 3.2 Potential Explanations

1. **File naming mismatch:** Files may exist with different names
2. **Location mismatch:** Files may be in different directories
3. **Generation pending:** Files may need to be generated from scripts
4. **Version control:** Files may be in different git branches or commits

---

## 4. Data Authenticity Concerns

### 4.1 Verified Authentic Data

✅ **De-knotting experiment** - Fully verified with matching CSV data
- Sample sizes: 200 runs per domain
- AUROC values match paper claims
- Delta calculations are correct
- Verdict interpretations are appropriate

### 4.2 Unverified Critical Claims

⚠️ **Main AUROC results** - Cannot verify without source data
- AoA values: 0.958, 0.799, 0.434
- Confidence intervals: [0.931, 0.980], [0.775, 0.822], [0.404, 0.464]
- Bootstrap methodology: B=10,000 resamplings

⚠️ **Reranking experiments** - Cannot verify without source data
- Best-of-N=64 pass@1 improvements
- Bootstrap confidence intervals
- Sample sizes: n=120 (math), n=198 (science), n=167 (coding)

⚠️ **Coding ablation studies** - Cannot verify without source data
- 83-scalar screen results
- Grouped feature ablation results
- CoT-only judge performance

---

## 5. Recommendations

### 5.1 Immediate Actions Required

1. **Locate or generate missing data files:**
   ```bash
   # Search for potentially renamed files
   find . -name "*reranking*" -o -name "*bootstrap*" -o -name "*ablation*"

   # Check for alternative locations
   ls -la results/tables/
   ls -la workshop/cotknot/results/tables/
   ```

2. **Verify sample size claims:**
   - Add explicit sample size breakdown to README
   - Cross-check with actual data files
   - Ensure consistency across all mentions

3. **Document data provenance:**
   - Add scripts used to generate each table
   - Include version information for data files
   - Document any preprocessing steps

### 5.2 Paper Revisions Needed

1. **Soften unverified claims** (if data cannot be located):
   - Change "achieves" to "reported as" or "approximately"
   - Remove precise confidence intervals if unverifiable
   - Add caveats about preliminary results

2. **Add data availability statements:**
   - Explicitly state which results are fully reproducible
   - Note any results that require external data access
   - Provide contact for data requests

3. **Improve transparency:**
   - Add exact file paths for all referenced data
   - Include checksums or file sizes for verification
   - Document any data cleaning or filtering steps

### 5.3 Repository Improvements

1. **Create data manifest:**
   ```markdown
   # Data Availability Manifest

   ## Fully Verified Results
   - [x] De-knotting experiments (deknot_alldomains_v2.csv)

   ## Pending Verification
   - [ ] Main AUROC results (aoa_bootstrap_ci.csv) - FILE NOT FOUND
   - [ ] Reranking experiments (bon_reranking_domain_pass1_ci.csv) - FILE NOT FOUND
   - [ ] Coding ablations (coding_feature_family_ablation.csv) - FILE NOT FOUND
   ```

2. **Add data generation scripts:**
   - Include scripts to regenerate all tables from raw data
   - Add requirements.txt for dependencies
   - Provide example usage in README

---

## 6. Conclusion

### Overall Assessment

**Consistency Score: 7/10**
- ✅ Paper and README numerical claims are consistent
- ⚠️ Critical data files are missing from repository
- ✅ Verified data (de-knotting) matches claims perfectly
- ❌ Cannot verify majority of experimental results

**Risk Level: MEDIUM-HIGH**

The core numerical claims are internally consistent between paper and README, and the one verified experimental result (de-knotting) matches perfectly. However, the absence of critical data files prevents full verification of most claims, which is a significant concern for academic rigor and reproducibility.

### Final Recommendation

**DO NOT SUBMIT** until:
1. All missing data files are located and verified, OR
2. Claims are softened to match available evidence, OR
3. Data availability issues are explicitly disclosed

**Priority Actions:**
1. Search repository history and branches for missing files
2. Contact original data collectors if files are truly missing
3. Regenerate data from source scripts if possible
4. Add comprehensive data documentation to repository

---

**Report Generated:** 2026-05-01
**Next Review:** After missing data files are located or claims are adjusted
