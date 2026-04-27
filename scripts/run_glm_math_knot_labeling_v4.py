"""
GLM Math Knot Labeling v4
=========================
Math-domain active-state knot annotation with a conservative v4 protocol.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
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


CONFIG = DOMAIN_CONFIGS["math"]
PROMPT_VERSION = "v4"
SYSTEM_PROMPT = (
    "You are an expert reader of olympiad-style math chain-of-thought traces. "
    "Detect only local active-state control breaks: moments where the prose loses "
    "stable control of the currently active assumption, case, symbol, subgoal, or "
    "tracked invariant. Be conservative and return valid JSON only."
)

USER_TEMPLATE = """\
A model is solving a math competition problem. Read its reasoning carefully.

[PROBLEM SNIPPET]
{problem_snippet}

[MODEL REASONING EXCERPTS (up to {max_chars} chars total)]
{think_text}

---
WHAT COUNTS AS A v4 KNOT
---
We are NOT scoring overall correctness, verbosity, or generic uncertainty.
A knot is a local active-state control break in the prose.

Mark knot_present="yes" only if ALL are true:
1. You can point to a short quote where the active math state becomes unstable.
2. The instability concerns an active assumption, case, symbol, subgoal, or invariant.
3. The instability is not repaired immediately in the next sentence, or later text keeps
   reasoning from the corrupted state.
4. The trace had already committed to one local setup, case, or symbol role before the instability.

If the trace is exploratory but still readable, answer "no".
If there is one clean self-correction and state is stable again immediately, answer "no".
If the best quote is only setup, plan narration, or unfinished algebra because the excerpt ends, answer "no".
If the model is still choosing coordinates, trying candidate lemmas, or wavering BEFORE committing to one setup, answer "no".
Prefer false negatives over false positives.

MATH SYMPTOMS:
1. case_split_instability — enters one case, then reasons with another case's facts without closure.
2. assumption_drift — an explicit condition or precondition is later silently weakened, flipped, or forgotten.
3. subgoal_frame_loss — local lemma / temporary target gets mixed up with the main goal.
4. variable_binding_drift — a symbol/object later takes a different role or identity in the same derivation.
5. invariant_drift — an explicitly tracked relation or invariant is later violated or dropped without notice.
6. repair_without_recovery — repeated repairs keep changing the same claim and do not restore one stable state.

NEGATIVE EXAMPLES:
- "Let A=(0,0), B=(1,0). Wait, that makes AB horizontal, good." -> no knot.
- "Assume n is odd. Then n=2k+1. Hence n^2 is odd." -> no knot.
- "Actually, I should expand before simplifying. Then x^2-1=(x-1)(x+1)." -> no knot.
- "Let me put D at (0,0)... or maybe rotate the rectangle first; suppose DC is horizontal for now." -> no knot unless later text uses two incompatible setups as simultaneously active.

POSITIVE EXAMPLES:
- "Assume n is even. Then write n=2k+1." -> case/assumption instability.
- "Let P be the midpoint of AB. ... Since P is a vertex of the triangle..." when P was never redefined -> variable binding drift.

Respond with EXACTLY these 11 keys:
{{
  "knot_present": "<yes|no>",
  "knot_severity": <0|1|2|3>,
  "knot_symptoms": ["<choose 0-3 from: case_split_instability, assumption_drift, subgoal_frame_loss, variable_binding_drift, invariant_drift, repair_without_recovery, none>"],
  "primary_trigger": "<case_split|assumption_tracking|subgoal_frame|symbol_binding|constraint_tracking|repair_loop|mixed|unclear|none>",
  "knot_quote": "<verbatim ≤25 words where the first knot appears, or empty string>",
  "trace_strategy": "<structural_abstraction|symbolic_manipulation|case_analysis|example_driven|patchy_backtracking>",
  "reversal_count": <integer>,
  "state_consistency": "<stable|minor_slip|recovered_break|lost_state|self_contradictory>",
  "recovers_later": "<yes|no>",
  "open_diagnosis": "<1-2 sentence concrete diagnosis>",
  "annotator_confidence": "<low|medium|high>"
}}

SEVERITY:
0 = no knot
1 = real but contained break, later recovered
2 = moderate break; active state unreliable for a span
3 = severe; state collapses into contradiction or unreadable control

FINAL CHECK:
- If your quote does not itself display the knot, answer "no".
- If a careful reader can still track the active case/symbol/subgoal clearly, answer "no".
- Do not call coordinate-search or lemma-search a knot unless a prior committed setup is later violated or reused incorrectly.
"""


def build_prompt(record: dict, think_max_chars: int) -> str:
    problem_snippet = str(record.get("actual_prompt") or record.get("prompt") or "")[:500]
    return USER_TEMPLATE.format(
        problem_snippet=problem_snippet,
        think_text=record["think_text"],
        max_chars=think_max_chars,
    )


def main():
    parser = argparse.ArgumentParser(description="GLM math knot labeling v4")
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

    print("Loading math reports ...")
    records, problem_keys, labels, problem_groups = load_runs_from_reports(
        CONFIG.report_paths,
        think_max_chars=args.think_max_chars,
    )
    print(f"  {len(records)} runs across {len(set(problem_keys))} problems from {len(CONFIG.report_paths)} datasets")

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
            problem_groups=problem_groups,
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

    dataset_counts = defaultdict(int)
    for problem_key in selected_problem_keys:
        dataset_counts[problem_groups.get(problem_key, "unknown")] += 1

    write_manifest(
        out_dir,
        {
            "mode": mode_name,
            "domain": CONFIG.domain,
            "datasets": sorted(CONFIG.report_paths),
            "protocol": "active_state_control_v4",
            "selected_runs": len(selected_records),
            "selected_problems": len(selected_problem_keys),
            "dataset_problem_counts": dict(dataset_counts),
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
    print("Next step: python scripts/analyze_glm_knot_v4.py --domain math")


if __name__ == "__main__":
    main()
