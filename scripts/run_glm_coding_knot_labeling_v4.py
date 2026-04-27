"""
GLM Coding Knot Labeling v4
===========================
Coding-domain active-state knot annotation with a conservative v4 protocol.
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


CONFIG = DOMAIN_CONFIGS["coding"]
PROMPT_VERSION = "v4"
SYSTEM_PROMPT = (
    "You are an expert reader of coding chain-of-thought traces. Detect only local "
    "active-state control breaks: moments where the prose loses stable control of the "
    "active specification, variable roles, branch state, loop state, or tracked invariant. "
    "Do not score generic uncertainty or mere code complexity. Return valid JSON only."
)

USER_TEMPLATE = """\
A model is solving a coding problem. Read its reasoning carefully.

[PROBLEM SNIPPET]
{problem_snippet}

[MODEL REASONING EXCERPTS (up to {max_chars} chars total)]
{think_text}

---
WHAT COUNTS AS A v4 KNOT
---
We are NOT scoring correctness, verbosity, or ordinary debugging language.
A knot is a local active-state control break in the coding prose.

Mark knot_present="yes" only if ALL are true:
1. A short quote visibly shows the active coding state becoming unstable.
2. The instability concerns the active spec, variable roles, branch/loop state,
   patch plan, or invariant.
3. The instability is not repaired immediately, or later text keeps reasoning from the corrupted state.
4. The trace had already committed to one active algorithmic interpretation before the instability.

Visible hard evidence for "yes" should be at least one of:
- explicit contradiction about the SAME requirement/state/branch,
- incompatible reuse of the SAME named variable or state object,
- explicit branch/loop invariant mix-up that downstream steps rely on,
- 3 or more repair turns on the SAME algorithmic plan with no stable resolution.

Healthy algorithm revision is NOT a knot.
A single clean off-by-one correction is NOT a knot.
Example interpretation noise before one settled plan is NOT a knot.
Prefer false negatives over false positives.

CODING SYMPTOMS:
1. spec_state_drift — after the active requirement is committed and used to design a step, later steps silently solve a different requirement and continue from it.
2. variable_role_swap — the same NAMED variable/index/container is reused with a different semantic role in the same active derivation, and later updates depend on the wrong role.
3. branch_entanglement — after entering one explicit case/branch, later text applies another branch's action or invariant to the SAME ongoing case without closing the first.
4. loop_state_forgetting — an explicit loop/window/pointer invariant is later used as if the iteration state were different, and the subsequent step depends on that mistaken state.
5. patch_backtracking — at least 3 repair turns keep replacing the algorithm skeleton without stabilizing one plan.
6. invariant_drop — an explicit ordering, complexity, parity, or state invariant is later violated or dropped without notice.

NEGATIVE EXAMPLES:
- "Wait, indices are 1-based in the statement, so subtract 1 once when reading." -> no knot.
- "I first thought BFS, but weighted edges mean Dijkstra; from here I only use Dijkstra." -> no knot.
- "Actually, for N=1 the answer is immediate; otherwise continue with the main loop." -> no knot.
- "The sample wording says 25 shifts, but the true cost comes from the weighted alphabet move definition." -> no knot if the later algorithm uses one consistent rule.
- "The statement allows either shift left or shift right." -> no knot by itself; merely restating both operations is not branch entanglement.
- "The minimal steps on a circle are min(clockwise, counterclockwise)." -> no knot unless later updates use two incompatible meanings of the same variable or state.
- Restating the sample, the allowed operations, or the problem constraint is not a knot unless later algorithm design keeps using the wrong active state.
- "The problem asks X. This is equivalent to Y." -> no knot unless later steps keep using both X and not-Y as active requirements.

POSITIVE EXAMPLES:
- "We need the next day >= d ... [later] so I should search strictly after d" with no acknowledged spec change -> spec state drift.
- "Let i be the left pointer ... [later] increment i as the number of chosen groups" in the same loop -> variable role swap.
- "If x is odd use branch A ... [later inside the same case] add the even-case transition" -> branch entanglement.
- "I will keep prefix sums fixed ... [later] update them in-place and still treat them as original" -> invariant drop.

Respond with EXACTLY these 11 keys:
{{
  "knot_present": "<yes|no>",
  "knot_severity": <0|1|2|3>,
  "knot_symptoms": ["<choose 0-3 from: spec_state_drift, variable_role_swap, branch_entanglement, loop_state_forgetting, patch_backtracking, invariant_drop, none>"],
  "primary_trigger": "<case_split|assumption_tracking|subgoal_frame|symbol_binding|constraint_tracking|repair_loop|mixed|unclear|none>",
  "knot_quote": "<verbatim ≤25 words where the first knot appears, or empty string>",
  "trace_strategy": "<plan_then_verify|local_trace|spec_then_patch|simulate_then_patch|patchy_backtracking>",
  "reversal_count": <integer>,
  "state_consistency": "<stable|minor_slip|recovered_break|lost_state|self_contradictory>",
  "recovers_later": "<yes|no>",
  "open_diagnosis": "<1-2 sentence concrete diagnosis>",
  "annotator_confidence": "<low|medium|high>"
}}

SEVERITY:
0 = no knot
1 = real but contained break, later recovered
2 = moderate break; active coding state unreliable for a span
3 = severe; contradictory or collapsed coding state

FINAL CHECK:
- If your quote does not itself show the instability, answer "no".
- If a careful reader can still track one coherent active plan/state, answer "no".
- Do not label sample parsing noise or one-shot clarification unless later algorithm design continues from the wrong active state.
- If the quote contains only statement restatement or sample debugging with no downstream polluted plan, answer "no".
- If you cannot point to an explicit contradiction, explicit incompatible variable/state reuse, or repeated failed plan repair, answer "no".
"""


def build_prompt(record: dict, think_max_chars: int) -> str:
    problem_snippet = str(record.get("actual_prompt") or record.get("prompt") or "")[:700]
    return USER_TEMPLATE.format(
        problem_snippet=problem_snippet,
        think_text=record["think_text"],
        max_chars=think_max_chars,
    )


def main():
    parser = argparse.ArgumentParser(description="GLM coding knot labeling v4")
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

    print("Loading LiveCodeBench coding report ...")
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
    print("Next step: python scripts/analyze_glm_knot_v4.py --domain coding")


if __name__ == "__main__":
    main()
