# GLM Knot Findings v4

This file summarizes the current v4 balanced-sample knot annotation outputs.

## Cross-domain summary

- **coding**: runs=256, accuracy=0.531, prevalence=0.087, rate_correct=0.096, rate_incorrect=0.076, mean_severity=0.177, rho(knot,correct)=0.037, rho(severity,correct)=0.038, degenerate=0
- **math**: runs=256, accuracy=0.528, prevalence=0.557, rate_correct=0.377, rate_incorrect=0.759, mean_severity=1.106, rho(knot,correct)=-0.384, rho(severity,correct)=-0.382, degenerate=0
- **science**: runs=256, accuracy=0.508, prevalence=0.268, rate_correct=0.252, rate_incorrect=0.285, mean_severity=0.536, rho(knot,correct)=-0.037, rho(severity,correct)=-0.037, degenerate=0

## Top symptoms by domain

### coding
- `variable_role_swap`: prevalence=0.087, rate_correct=0.096, rate_incorrect=0.076
- `branch_entanglement`: prevalence=0.079, rate_correct=0.089, rate_incorrect=0.067
- `spec_state_drift`: prevalence=0.055, rate_correct=0.067, rate_incorrect=0.042

### math
- `repair_without_recovery`: prevalence=0.520, rate_correct=0.346, rate_incorrect=0.716
- `variable_binding_drift`: prevalence=0.297, rate_correct=0.162, rate_incorrect=0.448
- `assumption_drift`: prevalence=0.256, rate_correct=0.215, rate_incorrect=0.302

### science
- `repair_without_recovery`: prevalence=0.216, rate_correct=0.173, rate_incorrect=0.260
- `mechanism_frame_swap`: prevalence=0.200, rate_correct=0.165, rate_incorrect=0.236
- `constraint_mismatch`: prevalence=0.152, rate_correct=0.173, rate_incorrect=0.130
