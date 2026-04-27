"""
GLM Science Knot Labeling v4
============================
Science-domain active-state knot annotation with a conservative v4 protocol.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from knot_glm_common import (
    GLM_MODEL,
    all_eligible_runs,
    execute_labeling,
    get_api_key,
    load_runs_from_reports,
    recommend_workers,
    sample_runs,
    write_manifest,
)
from knot_v4_configs import DOMAIN_CONFIGS


CONFIG = DOMAIN_CONFIGS["science"]
PROMPT_VERSION = "v4"
SYSTEM_PROMPT = (
    "You are an expert reader of graduate-level science chain-of-thought traces "
    "(physics, chemistry, biology). Detect only local active-state control breaks: "
    "moments where the prose loses stable control of the active entity, mechanism, "
    "constraint, evidential reading, or tracked physical quantity. Return valid JSON only."
)

USER_TEMPLATE = """\
A model is answering a graduate-level science multiple-choice question (GPQA).
Read its reasoning carefully.

[PROBLEM SNIPPET]
{problem_snippet}

[MODEL REASONING EXCERPTS (up to {max_chars} chars total)]
{think_text}

---
WHAT COUNTS AS A v4 KNOT
---
We are NOT scoring correctness, length, or generic uncertainty.
A knot is a local active-state control break in the scientific prose.

Mark knot_present="yes" only if ALL are true:
1. A short quote visibly shows the active scientific state becoming unstable.
2. The instability concerns entity identity, mechanism frame, constraint use,
   evidence reading, unit/scale interpretation, or repeated failed repair.
3. The instability is not repaired immediately, or later text keeps reasoning from the corrupted state.
4. The trace had already committed to one active mechanism, identity, or constraint before the instability.

Visible hard evidence for "yes" should be at least one of:
- explicit contradiction about the SAME entity/result/regime,
- incompatible reuse of the SAME symbol/entity after commitment,
- 3 or more repair turns on the SAME claim with no stable resolution,
- explicit use of a formula/result the trace itself already said was not applicable.

Normal hypothesis testing is NOT a knot.
A single clean self-correction is NOT a knot.
Mere uncertainty before commitment is NOT a knot.
Prefer false negatives over false positives.

SCIENCE SYMPTOMS:
1. entity_binding_drift — after one entity/symbol identity is committed, later text reuses it as a different entity in the same active derivation.
2. mechanism_frame_swap — after committing to mechanism/pathway A, later text COMPUTES or SELECTS the same intermediate/result using incompatible pathway B without closing A.
3. constraint_mismatch — an explicit regime, law, or constraint is later violated or replaced without acknowledgement after it had become the active basis for a calculation.
4. evidence_misread — a stated observation (spectrum, sign, units, counts, options) is first treated as evidence E, then later used as if it were incompatible evidence E'.
5. unit_scale_slip — active numeric scale/unit interpretation changes and contaminates later reasoning.
6. repair_without_recovery — at least 3 repair turns keep changing the same scientific claim without restoring one stable state.

NEGATIVE EXAMPLES:
- "Could be SN1, but primary substrate makes SN2 more likely, so choose SN2." -> no knot.
- "Wait, 298×0.08=23.84, so ΔG≈6.2." -> no knot.
- "Carbonyl at 1715 cm^-1 suggests ketone; no aldehyde proton, so ketone." -> no knot.
- "Maybe DSG means X ... no, in this context it likely means Y, and from here only Y is used." -> no knot.
- "Some sites may be transient or context-dependent." -> no knot unless the text later uses two incompatible active interpretations of the same site.
- "Perhaps the ether undergoes electrophilic addition or cleavage." -> no knot by itself; branching hypotheses before a settled mechanism are not enough.
- "PFA is paraformaldehyde ... DSG is likely ..." -> no knot unless later steps keep using both incompatible identities.
- "Perhaps A is a meson or something." -> no knot unless later calculations also treat A as a different settled particle identity.

POSITIVE EXAMPLES:
- "Under the weak-field approximation ... [later] use strong-field splitting for the same calculation with no regime switch." -> constraint mismatch.
- "Let E be the electric field ... [later] E = kinetic energy in the same running derivation." -> entity binding drift.
- "The product must be an alcohol ... [later] the same product must be an alkyne, still using it as the same intermediate." -> mechanism frame / repair failure.

Respond with EXACTLY these 11 keys:
{{
  "knot_present": "<yes|no>",
  "knot_severity": <0|1|2|3>,
  "knot_symptoms": ["<choose 0-3 from: entity_binding_drift, mechanism_frame_swap, constraint_mismatch, evidence_misread, unit_scale_slip, repair_without_recovery, none>"],
  "primary_trigger": "<case_split|assumption_tracking|subgoal_frame|symbol_binding|constraint_tracking|repair_loop|mixed|unclear|none>",
  "knot_quote": "<verbatim ≤25 words where the first knot appears, or empty string>",
  "trace_strategy": "<mechanism_tracing|symbolic_manipulation|case_analysis|evidence_matching|patchy_backtracking>",
  "reversal_count": <integer>,
  "state_consistency": "<stable|minor_slip|recovered_break|lost_state|self_contradictory>",
  "recovers_later": "<yes|no>",
  "open_diagnosis": "<1-2 sentence concrete diagnosis>",
  "annotator_confidence": "<low|medium|high>"
}}

SEVERITY:
0 = no knot
1 = real but contained break, later recovered
2 = moderate break; active scientific state unreliable for a span
3 = severe; contradictory or collapsed scientific state

FINAL CHECK:
- If your quote does not itself display the instability, answer "no".
- If a careful reader can still track one coherent active scientific state, answer "no".
- Do not label generic mechanistic brainstorming or reagent-identity guessing unless the trace later reasons from two incompatible committed states.
- If the quote contains only "maybe / perhaps / could" exploration with no downstream polluted calculation, answer "no".
- If you cannot point to an explicit contradiction, explicit incompatible reuse, or repeated failed repair, answer "no".
"""


def build_prompt(record: dict, think_max_chars: int) -> str:
    problem_snippet = str(record.get("actual_prompt") or record.get("prompt") or "")[:500]
    return USER_TEMPLATE.format(
        problem_snippet=problem_snippet,
        think_text=record["think_text"],
        max_chars=think_max_chars,
    )


def main():
    parser = argparse.ArgumentParser(description="GLM science knot labeling v4")
    parser.add_argument("--n-problems", type=int, default=CONFIG.default_n_problems)
    parser.add_argument("--runs-per-problem", type=int, default=CONFIG.default_runs_per_problem)
    parser.add_argument("--all-eligible", action="store_true")
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--requests-per-second", type=float, default=CONFIG.default_requests_per_second)
    parser.add_argument("--model", type=str, default=GLM_MODEL)
    parser.add_argument("--think-max-chars", type=int, default=16000)
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--out-dir", type=str, default=str(CONFIG.default_out_dir))
    args = parser.parse_args()

    api_key = get_api_key(args.api_key)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading GPQA science report ...")
    records, problem_keys, labels, _ = load_runs_from_reports(
        CONFIG.report_paths,
        think_max_chars=args.think_max_chars,
    )
    print(f"  {len(records)} runs across {len(set(problem_keys))} problems")

    if args.all_eligible:
        selected_indices, selected_problem_keys = all_eligible_runs(problem_keys, labels, seed=args.seed)
        mode_name = "all_eligible"
    else:
        selected_indices, selected_problem_keys = sample_runs(
            records,
            problem_keys,
            labels,
            n_problems=args.n_problems,
            runs_per_problem=args.runs_per_problem,
            seed=args.seed,
            balance_groups=CONFIG.balance_groups,
        )
        mode_name = "sample"

    if args.max_runs > 0:
        selected_indices = selected_indices[: args.max_runs]

    selected_records = [records[idx] for idx in selected_indices]
    print(f"Selected {len(selected_records)} runs across {len(selected_problem_keys)} problems ({mode_name})")

    workers, cpu_info = recommend_workers()
    if args.workers > 0:
        workers = args.workers
    print("CPU / load check:")
    print(json.dumps(cpu_info, ensure_ascii=False))

    write_manifest(
        out_dir,
        {
            "mode": mode_name,
            "domain": CONFIG.domain,
            "datasets": sorted(CONFIG.report_paths),
            "protocol": "active_state_control_v4",
            "selected_runs": len(selected_records),
            "selected_problems": len(selected_problem_keys),
            "workers": workers,
            "requests_per_second": args.requests_per_second,
            "model": args.model,
            "think_max_chars": args.think_max_chars,
            "prompt_version": PROMPT_VERSION,
            "seed": args.seed,
            "cpu_info": cpu_info,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )

    counts = execute_labeling(
        selected_records=selected_records,
        domain=CONFIG.domain,
        prompt_version=PROMPT_VERSION,
        api_key=api_key,
        system_prompt=SYSTEM_PROMPT,
        prompt_builder=build_prompt,
        out_dir=out_dir,
        workers=workers,
        requests_per_second=args.requests_per_second,
        model=args.model,
        think_max_chars=args.think_max_chars,
        progress_every=25,
    )
    print(f"\nDone. Status counts: {counts}")
    print(f"Raw JSONs in: {out_dir}")
    print("Next step: python scripts/analyze_glm_knot_v4.py --domain science")


if __name__ == "__main__":
    main()
