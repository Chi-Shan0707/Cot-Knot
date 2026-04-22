"""
GLM Math Knot Span Extraction
=============================
Use GLM to extract only explicit math reasoning-knot spans from CoT excerpts.
Unlike the run-level classifier, this prompt allows zero spans and is tuned for
high precision.

Output directory:
  results/glm_math_knot_span_raw_v1/
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import re
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_glm_math_knot_labeling import (
    AdaptivePacer,
    GLM_ENDPOINT,
    GLM_MODEL,
    MAX_RETRIES,
    BASE_RETRY_DELAY,
    RateLimiter,
    all_eligible_runs,
    get_api_key,
    load_math_runs,
    recommend_workers,
    sample_runs,
)

PROMPT_VERSION = "span_v1"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "glm_math_knot_span_raw_v1"


SYSTEM_PROMPT = (
    "You are an expert at reading olympiad-style math chain-of-thought traces. "
    "Extract only explicit local reasoning-knot spans. A knot span must show a "
    "concrete local state break in the text itself. If no explicit span is visible, "
    "return an empty list. Respond in valid JSON only."
)


USER_TEMPLATE = """\
A model is solving a math competition problem. Below are excerpts from its reasoning trace.

[PROBLEM SNIPPET]
{problem_snippet}

[MODEL REASONING EXCERPTS]
{think_text}

We are NOT scoring overall quality or correctness. We only want explicit local knot spans.

Return a JSON object with EXACTLY these 3 keys:
{{
  "explicit_knot_spans": [
    {{
      "quote": "<verbatim ≤30 words from the trace>",
      "symptom": "<case_split_instability|assumption_drift|subgoal_frame_loss|variable_binding_drift|invariant_drift|repair_without_recovery>",
      "trigger": "<case_split|assumption_tracking|subgoal_frame|symbol_binding|mixed|unclear>",
      "why_it_is_a_knot": "<one-sentence explanation>"
    }}
  ],
  "trace_strategy": "<structural_abstraction|symbolic_manipulation|case_analysis|example_driven|patchy_backtracking>",
  "overall_state": "<stable|minor_slip|lost_state|self_contradictory|not_applicable>"
}}

HARD RULES:
- Return at most 2 spans.
- Return "explicit_knot_spans": [] if no explicit span is visible.
- Only extract a span if the quote itself shows one of:
  1) explicit contradiction,
  2) incompatible redefinition of the same symbol/object,
  3) switching cases without closing the previous case,
  4) repeated repair that still leaves the active state unclear.
- Coordinate setup, variable introduction, sanity checks, restating equations, or trying a new method are NOT enough.
- If the quote is readable as a normal self-check, do NOT extract it.
- Do not infer from omitted parts of the trace.

NEGATIVE EXAMPLES — do NOT extract:
1) "Let A=(0,0), B=(107,0), C=(107,16), D=(0,16). Wait, DC has length 107, so this setup is consistent."
2) "Let N = 1000A + 100B + 10C + D. So N is the original number."

POSITIVE EXAMPLES — extract:
1) "Assume n is even. Then n=2k+1."
2) "In Case 1, x>0. Therefore x<0 here."

Prefer false negatives over false positives.
"""


def build_prompt(record: dict) -> str:
    problem_snippet = str(record.get("actual_prompt") or record.get("prompt") or "")[:500]
    return USER_TEMPLATE.format(
        problem_snippet=problem_snippet,
        think_text=record["think_text"],
    )


def call_glm(prompt_text: str, api_key: str, model: str) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ],
        "temperature": 0.1,
        "max_tokens": 900,
    }
    for attempt in range(1, MAX_RETRIES + 1):
        delay = BASE_RETRY_DELAY * (2 ** (attempt - 1))
        try:
            resp = requests.post(GLM_ENDPOINT, headers=headers, json=payload, timeout=90)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as error:
            status = error.response.status_code if error.response is not None else "?"
            print(f"  HTTP {status} (attempt {attempt}/{MAX_RETRIES}): {error}")
            if status in (401, 403):
                return {}
        except requests.Timeout:
            print(f"  Timeout (attempt {attempt}/{MAX_RETRIES})")
        except requests.RequestException as error:
            print(f"  Request error (attempt {attempt}/{MAX_RETRIES}): {error}")
        if attempt < MAX_RETRIES:
            time.sleep(delay)
    return {}


def extract_content(glm_response: dict) -> str:
    try:
        return glm_response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""


def parse_glm_content(content: str) -> tuple[dict, str | None]:
    clean = re.sub(r"```[a-zA-Z]*\n?", "", str(content)).replace("```", "").strip()
    if not clean.startswith("{"):
        start = clean.find("{")
        end = clean.rfind("}")
        if start >= 0 and end > start:
            clean = clean[start : end + 1]
    try:
        return json.loads(clean), None
    except json.JSONDecodeError as error:
        return {}, str(error)


def out_path(out_dir: Path, record: dict) -> Path:
    safe_problem = str(record["problem_id"]).replace("/", "_")
    return out_dir / f"{record['dataset']}__{safe_problem}__run{int(record['run_index']):05d}.json"


def already_done(out_dir: Path, record: dict) -> bool:
    return out_path(out_dir, record).exists()


def save_result(out_dir: Path, record: dict, glm_content: str):
    parsed, parse_error = parse_glm_content(glm_content)
    payload = {
        "dataset": record["dataset"],
        "problem_id": record["problem_id"],
        "problem_key": record["problem_key"],
        "run_index": int(record["run_index"]),
        "is_correct": int(record["is_correct"]),
        "think_chars": len(record["think_text"]),
        "think_total_chars": int(record.get("think_total_chars", len(record["think_text"]))),
        "prompt_version": PROMPT_VERSION,
        "glm_raw_content": glm_content,
        "glm_parsed": parsed,
        "parse_error": parse_error,
    }
    with open(out_path(out_dir, record), "w") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def consolidate_to_jsonl(out_dir: Path) -> Path:
    jsonl_path = out_dir / "all_runs.jsonl"
    rows = []
    for path in sorted(out_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        try:
            rows.append(json.loads(path.read_text()))
        except Exception:
            pass
    with open(jsonl_path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Consolidated {len(rows)} records -> {jsonl_path}")
    return jsonl_path


def run_one(record: dict, api_key: str, out_dir: Path, rate_limiter: RateLimiter, model: str) -> dict:
    if already_done(out_dir, record):
        return {"status": "skip"}
    rate_limiter.wait()
    glm_raw = call_glm(build_prompt(record), api_key, model=model)
    glm_content = extract_content(glm_raw)
    if not glm_content:
        return {"status": "fail"}
    save_result(out_dir, record, glm_content)
    parsed, parse_error = parse_glm_content(glm_content)
    if parse_error:
        return {"status": "parse_error"}
    spans = parsed.get("explicit_knot_spans", [])
    return {"status": "ok", "n_spans": len(spans) if isinstance(spans, list) else -1}


def main():
    parser = argparse.ArgumentParser(description="GLM math knot span extraction")
    parser.add_argument("--datasets", nargs="+", default=["aime24", "aime25", "brumo25", "hmmt25"])
    parser.add_argument("--n-problems", type=int, default=48)
    parser.add_argument("--runs-per-problem", type=int, default=4)
    parser.add_argument("--all-eligible", action="store_true")
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--requests-per-second", type=float, default=2.0)
    parser.add_argument("--model", type=str, default=GLM_MODEL)
    parser.add_argument("--think-max-chars", type=int, default=16000)
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    api_key = get_api_key(args.api_key)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading math reports ...")
    records, problem_keys, labels = load_math_runs(args.datasets, think_max_chars=args.think_max_chars)
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
        )
        mode_name = "sample"

    if args.max_runs > 0:
        selected_indices = selected_indices[: args.max_runs]

    selected_records = [records[idx] for idx in selected_indices]
    print(f"Selected {len(selected_records)} runs across {len(selected_problem_keys)} problems ({mode_name})")

    workers, cpu_info = recommend_workers()
    if args.workers > 0:
        workers = args.workers
    print(json.dumps(cpu_info, ensure_ascii=False))

    rate_limiter = RateLimiter(args.requests_per_second)
    counts = defaultdict(int)
    completed = 0

    with futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(run_one, record, api_key, out_dir, rate_limiter, args.model): record
            for record in selected_records
        }
        for future in futures.as_completed(future_map):
            result = future.result()
            counts[result["status"]] += 1
            completed += 1
            if completed % 25 == 0 or result["status"] in {"fail", "parse_error"}:
                print(
                    f"[{completed:5d}/{len(selected_records)}] "
                    f"ok={counts['ok']} skip={counts['skip']} "
                    f"parse={counts['parse_error']} fail={counts['fail']}"
                )

    consolidate_to_jsonl(out_dir)
    print(f"Done. Status counts: {dict(counts)}")


if __name__ == "__main__":
    main()
