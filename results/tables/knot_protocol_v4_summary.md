# Knot Protocol v4

## Core rule

- A `knot` is a local active-state control break in the reasoning prose.
- Mark `yes` only when the visible text itself shows state instability and that instability is not repaired immediately.
- Do **not** mark generic uncertainty, clean one-shot self-correction, or normal exploration.

## Shared output schema

- `knot_present`
- `knot_severity`
- `knot_symptoms`
- `primary_trigger`
- `knot_quote`
- `trace_strategy`
- `reversal_count`
- `state_consistency`
- `recovers_later`
- `open_diagnosis`
- `annotator_confidence`

## Reporting note

- Raw GLM outputs are preserved in the raw JSONL dumps.
- Final `v4` CSV summaries use a conservative protocol filter on top of raw positives.
- For science and coding in particular, treat the filtered CSV labels/summaries as the main protocol outputs, not the raw positive rate.

## Domain symptom banks

### Math

- `case_split_instability`
- `assumption_drift`
- `subgoal_frame_loss`
- `variable_binding_drift`
- `invariant_drift`
- `repair_without_recovery`

### Science

- `entity_binding_drift`
- `mechanism_frame_swap`
- `constraint_mismatch`
- `evidence_misread`
- `unit_scale_slip`
- `repair_without_recovery`

### Coding

- `spec_state_drift`
- `variable_role_swap`
- `branch_entanglement`
- `loop_state_forgetting`
- `patch_backtracking`
- `invariant_drop`
