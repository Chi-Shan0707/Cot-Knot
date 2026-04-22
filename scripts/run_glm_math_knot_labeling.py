"""
GLM Math Knot Labeling
======================
Read math chain-of-thought traces from evaluation reports and ask GLM to label
local "reasoning knots": places where the prose loses stable control of active
assumptions, symbols, subgoals, or tracked constraints.

This is intentionally analogous to the coding-side knot labeling, but the schema
is math-specific. It does NOT ask for generic hesitation. It asks for concrete
local instability in proof/search state.

Outputs:
  results/glm_math_knot_raw_v1/
    - one JSON per labeled run
    - all_runs.jsonl
    - _manifest.json
    - _failed_runs.jsonl

Usage:
  GLM_API_KEY=<key> python scripts/run_glm_math_knot_labeling.py --max-runs 16
  GLM_API_KEY=<key> python scripts/run_glm_math_knot_labeling.py --all-eligible
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import re
import sys
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import requests


REPO_ROOT = Path(__file__).resolve().parent.parent
NAD_ROOT = REPO_ROOT.parent / "NAD_Next"
PROMPT_VERSION = "v3"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "glm_math_knot_raw_v3"

MATH_REPORTS = {
    "aime24": NAD_ROOT / "MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/aime24/cache_neuron_output_1_act_no_rms_20250902_025610/evaluation_report.json",
    "aime25": NAD_ROOT / "MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/aime25/cache_neuron_output_1_act_no_rms_20251126_114548/evaluation_report.json",
    "brumo25": NAD_ROOT / "MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/brumo25/cache_neuron_output_1_act_no_rms_20251126_071142/evaluation_report.json",
    "hmmt25": NAD_ROOT / "MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/hmmt25/cache_neuron_output_1_act_no_rms_20251126_223151/evaluation_report.json",
}

GLM_ENDPOINT = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
GLM_MODEL = "glm-4-flash"
MAX_RETRIES = 5
BASE_RETRY_DELAY = 5

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def get_api_key(cli_key: str = "") -> str:
    key = cli_key or os.environ.get("GLM_API_KEY", "")
    if not key:
        sys.exit("ERROR: Set GLM_API_KEY env var or pass --api-key.")
    return key


class AdaptivePacer:
    def __init__(self, window=10, fail_thresh=0.5, sleep_min=0.5, sleep_max=30.0):
        self.window = window
        self.fail_thresh = fail_thresh
        self.sleep_min = sleep_min
        self.sleep_max = sleep_max
        self._history: deque = deque(maxlen=window)
        self._current_sleep = sleep_min

    def record(self, success: bool):
        self._history.append(success)
        if len(self._history) < self.window:
            return
        fail_rate = self._history.count(False) / len(self._history)
        if fail_rate > self.fail_thresh:
            self._current_sleep = min(self._current_sleep * 2, self.sleep_max)
            print(f"\n  High failure rate ({fail_rate:.0%}). Slowing to {self._current_sleep:.1f}s pause.")
        elif fail_rate == 0.0 and self._current_sleep > self.sleep_min:
            self._current_sleep = max(self._current_sleep / 1.5, self.sleep_min)

    def sleep(self):
        time.sleep(self._current_sleep)


class RateLimiter:
    def __init__(self, requests_per_second: float):
        self.interval = 0.0 if requests_per_second <= 0 else 1.0 / requests_per_second
        self.lock = threading.Lock()
        self.next_ready = 0.0

    def wait(self):
        if self.interval <= 0:
            return
        sleep_for = 0.0
        with self.lock:
            now = time.monotonic()
            if now < self.next_ready:
                sleep_for = self.next_ready - now
                self.next_ready += self.interval
            else:
                self.next_ready = now + self.interval
        if sleep_for > 0:
            time.sleep(sleep_for)


def recommend_workers() -> tuple[int, dict]:
    logical = os.cpu_count() or 4
    try:
        load1, load5, load15 = os.getloadavg()
    except OSError:
        load1, load5, load15 = 0.0, 0.0, 0.0
    load_ratio = load1 / max(logical, 1)
    if load_ratio >= 0.7:
        workers = 2
    elif load_ratio >= 0.35:
        workers = 4
    else:
        workers = min(8, max(2, logical // 32))
    info = {
        "logical_cpus": logical,
        "loadavg_1m": round(load1, 3),
        "loadavg_5m": round(load5, 3),
        "loadavg_15m": round(load15, 3),
        "load_ratio_1m": round(load_ratio, 4),
    }
    return workers, info


def extract_think(generated_text: str) -> str:
    match = THINK_RE.search(generated_text)
    return match.group(1).strip() if match else generated_text


def _bounded_window(total_chars: int, center: int, width: int) -> tuple[int, int]:
    start = max(0, center - width // 2)
    end = min(total_chars, start + width)
    start = max(0, end - width)
    return start, end


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ranges:
        return []
    ordered = sorted(ranges)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def build_trace_excerpt(generated_text: str, max_chars: int = 16000) -> tuple[str, int]:
    think_text = extract_think(generated_text)
    total_chars = len(think_text)
    if total_chars <= max_chars:
        return think_text, total_chars

    separator_budget = 500
    window_count = 4
    window_chars = max(1800, (max_chars - separator_budget) // window_count)
    anchor_centers = [
        window_chars // 2,
        total_chars // 3,
        (2 * total_chars) // 3,
        total_chars - window_chars // 2,
    ]
    ranges = [_bounded_window(total_chars, center, window_chars) for center in anchor_centers]
    merged = _merge_ranges(ranges)

    slices = []
    for idx, (start, end) in enumerate(merged, 1):
        pct_start = int(round(100 * start / total_chars))
        pct_end = int(round(100 * end / total_chars))
        header = f"[TRACE SLICE {idx}: {pct_start}%–{pct_end}%]"
        body = think_text[start:end].strip()
        slices.append(f"{header}\n{body}")
    excerpt = "\n\n...\n\n".join(slices)
    return excerpt[: max_chars + 400], total_chars


def load_math_runs(datasets: list[str], think_max_chars: int) -> tuple[list[dict], list[str], list[int]]:
    records: list[dict] = []
    problem_keys: list[str] = []
    labels: list[int] = []

    for dataset in datasets:
        report_path = MATH_REPORTS[dataset]
        report = json.loads(report_path.read_text())
        for problem in report["results"]:
            problem_id = str(problem["problem_id"])
            prompt = str(problem.get("prompt") or "")
            runs = problem.get("runs", [])
            for run in runs:
                generated_text = str(run.get("generated_text") or "")
                think_excerpt, think_total_chars = build_trace_excerpt(
                    generated_text,
                    max_chars=think_max_chars,
                )
                record = {
                    "dataset": dataset,
                    "problem_id": problem_id,
                    "problem_key": f"{dataset}:{problem_id}",
                    "run_index": int(run["run_index"]),
                    "is_correct": int(bool(run.get("is_correct", False))),
                    "prompt": prompt,
                    "actual_prompt": str(run.get("actual_prompt") or ""),
                    "generated_text": generated_text,
                    "think_text": think_excerpt,
                    "think_total_chars": think_total_chars,
                    "source_report": str(report_path),
                }
                records.append(record)
                problem_keys.append(record["problem_key"])
                labels.append(record["is_correct"])

    return records, problem_keys, labels


def _stratified_problem_sample(problem_acc: dict[str, float], candidate_pids: list[str], n_select: int) -> list[str]:
    if n_select <= 0:
        return []
    if len(candidate_pids) <= n_select:
        return list(candidate_pids)

    ordered = sorted(candidate_pids, key=lambda pid: problem_acc[pid])
    n_hard = n_select // 3
    n_easy = n_select // 3
    n_mid = n_select - n_hard - n_easy

    chosen: list[str] = []
    chosen.extend(ordered[:n_hard])
    if n_easy > 0:
        chosen.extend(ordered[-n_easy:])

    chosen_set = set(chosen)
    mid_pool = [pid for pid in ordered if pid not in chosen_set]
    mid_pool = sorted(mid_pool, key=lambda pid: abs(problem_acc[pid] - 0.5))
    chosen.extend(mid_pool[:n_mid])

    if len(chosen) < n_select:
        remainder = [pid for pid in ordered if pid not in set(chosen)]
        chosen.extend(remainder[: n_select - len(chosen)])

    return chosen[:n_select]


def sample_runs(
    records: list[dict],
    problem_keys: list[str],
    labels: list[int],
    n_problems: int = 48,
    runs_per_problem: int = 4,
    seed: int = 42,
) -> tuple[list[int], list[str]]:
    by_problem: dict[str, list[tuple[int, int]]] = defaultdict(list)
    by_dataset: dict[str, list[str]] = defaultdict(list)
    for rec_idx, (problem_key, label) in enumerate(zip(problem_keys, labels)):
        by_problem[problem_key].append((rec_idx, int(label)))
        dataset = problem_key.split(":", 1)[0]
        if problem_key not in by_dataset[dataset]:
            by_dataset[dataset].append(problem_key)

    eligible_by_dataset: dict[str, list[str]] = defaultdict(list)
    problem_acc: dict[str, float] = {}
    for problem_key, items in by_problem.items():
        if not any(label == 1 for _, label in items):
            continue
        if not any(label == 0 for _, label in items):
            continue
        dataset = problem_key.split(":", 1)[0]
        eligible_by_dataset[dataset].append(problem_key)
        problem_acc[problem_key] = sum(label for _, label in items) / len(items)

    active_datasets = [dataset for dataset in MATH_REPORTS if eligible_by_dataset.get(dataset)]
    if not active_datasets:
        return [], []

    base = n_problems // len(active_datasets)
    remainder = n_problems % len(active_datasets)
    selected_problem_keys: list[str] = []
    for idx, dataset in enumerate(active_datasets):
        need = base + (1 if idx < remainder else 0)
        selected_problem_keys.extend(
            _stratified_problem_sample(problem_acc, eligible_by_dataset[dataset], need)
        )

    selected_problem_keys = selected_problem_keys[:n_problems]

    rng = np.random.default_rng(seed)
    selected_indices: list[int] = []
    for problem_key in selected_problem_keys:
        items = by_problem[problem_key]
        correct_idx = [rec_idx for rec_idx, label in items if label == 1]
        incorrect_idx = [rec_idx for rec_idx, label in items if label == 0]
        rng.shuffle(correct_idx)
        rng.shuffle(incorrect_idx)
        half = runs_per_problem // 2
        chosen = correct_idx[:half] + incorrect_idx[:half]
        if len(chosen) < runs_per_problem:
            all_idx = correct_idx + incorrect_idx
            rng.shuffle(all_idx)
            for rec_idx in all_idx:
                if rec_idx not in chosen:
                    chosen.append(rec_idx)
                if len(chosen) >= runs_per_problem:
                    break
        selected_indices.extend(chosen)

    return selected_indices, selected_problem_keys


def all_eligible_runs(problem_keys: list[str], labels: list[int], seed: int = 42) -> tuple[list[int], list[str]]:
    by_problem: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for rec_idx, (problem_key, label) in enumerate(zip(problem_keys, labels)):
        by_problem[problem_key].append((rec_idx, int(label)))

    selected_indices: list[int] = []
    selected_problem_keys: list[str] = []
    for problem_key in sorted(by_problem):
        items = by_problem[problem_key]
        if not any(label == 1 for _, label in items):
            continue
        if not any(label == 0 for _, label in items):
            continue
        selected_problem_keys.append(problem_key)
        selected_indices.extend(rec_idx for rec_idx, _ in items)

    rng = np.random.default_rng(seed)
    rng.shuffle(selected_indices)
    return selected_indices, selected_problem_keys


SYSTEM_PROMPT = (
    "You are an expert at reading olympiad-style math chain-of-thought traces. "
    "Your task is to detect local reasoning-knot points: places where the prose "
    "loses stable control of the currently active assumption, case, symbol, subgoal, "
    "or tracked quantity. Do not score generic hesitation or overall quality. "
    "Respond in valid JSON only."
)


USER_TEMPLATE = """\
A model is solving a math competition problem. Read its chain-of-thought reasoning carefully.

[PROBLEM SNIPPET]
{problem_snippet}

[MODEL REASONING EXCERPTS (up to {max_chars} chars total)]
{think_text}

---
BACKGROUND — what to detect
---
The displayed reasoning may be a set of non-contiguous slices from a longer trace.
Only use what is explicitly visible in the shown text. Do not infer a knot from
omitted regions, unfinished setup, or simple truncation.

We are NOT scoring correctness, confidence, or generic uncertainty words.
We are looking for local reasoning knots: places where the text itself loses a
stable proof/search state. A knot is present only when a careful reader can point
to a short quote showing that the currently active assumption, case, symbol,
subgoal, or tracked quantity has become unstable or contradictory.

CONSERVATIVE LABELING RULE:
- Only answer knot_present="yes" if you can justify it with a short verbatim quote
  from the reasoning itself.
- If the reasoning is exploratory but state remains interpretable, answer "no".
- A single clean self-correction is healthy if it restores a stable state.
- Being wrong is not enough. The issue must be local state instability in the prose.
- If the best quote is only coordinate setup, variable introduction, or plan
  exploration, answer "no".
- If the quote does not itself contain the unstable moment, answer "no".
- If the instability resolves within the next sentence and the state is readable
  again, answer "no".
- To answer "yes", the excerpt should show one of these concrete evidence forms:
  (a) explicit contradiction, (b) incompatible redefinition of the same object,
  (c) case switch without closing the prior case, or (d) repeated repair that
  still leaves the active state unclear.

CORE MATH KNOT SYMPTOMS:

1. case_split_instability
   The trace enters one case / assumption, then silently reasons with facts from a
   different case or forgets which case is currently active.

2. assumption_drift
   A condition, bound, theorem precondition, or temporary assumption is introduced,
   then later weakened, forgotten, flipped, or reused incorrectly.

3. subgoal_frame_loss
   The model switches between a local lemma / temporary target and the main goal
   without a clean handoff, or treats a local result as if the main goal were proved.

4. variable_binding_drift
   A previously defined symbol, point, variable, index, or quantified object later
   gets used with a different role, identity, or scope.

5. invariant_drift
   The model states a relation / invariant / conserved quantity to track, then later
   violates or ignores that same tracked object without noticing.

6. repair_without_recovery
   Repeated patches like "wait / actually / no" try to repair the argument, but do
   not restore a stable active state. Use this only when there are repeated repairs
   or an explicit contradiction and the later text remains unstable.

IMPORTANT:
- Do not label healthy case analysis as case_split_instability.
- Do not label normal variable substitution as variable_binding_drift.
- Do not label short uncertainty phrases by themselves.
- Do not label unfinished coordinate setup or unfinished algebra because the excerpt ends.
- A run can be wrong but still have knot_present="no" if its local state remains readable.

NEGATIVE EXAMPLE (answer "no"):
"Let A=(0,0), B=(107,0), C=(107,16), D=(0,16). Wait, DC has length 107, so this setup is consistent."
Reason: this is a clean check of a coordinate assignment, not a persistent knot.

POSITIVE EXAMPLE (answer "yes"):
"Assume n is even. Then write n=2k+1. Hence n is odd in this case."
Reason: the active assumption is contradicted without a clean handoff.

---
TASK
---
Respond with a JSON object containing EXACTLY these 9 keys:

{{
  "knot_present": "<yes|no>",
  "knot_severity": <0|1|2|3>,
  "knot_symptoms": ["<choose 0-3 from: case_split_instability, assumption_drift, subgoal_frame_loss, variable_binding_drift, invariant_drift, repair_without_recovery, none>"],
  "primary_trigger": "<case_split|assumption_tracking|subgoal_frame|symbol_binding|mixed|unclear|none>",
  "knot_quote": "<verbatim ≤25 words from the reasoning where the first knot appears, or empty string>",
  "trace_strategy": "<structural_abstraction|symbolic_manipulation|case_analysis|example_driven|patchy_backtracking>",
  "reversal_count": <integer, count of explicit repair turns like wait/actually/no/but>,
  "state_consistency": "<stable|minor_slip|lost_state|self_contradictory|not_applicable>",
  "open_diagnosis": "<1-2 sentence concrete diagnosis of the knot or why the trace stays stable>"
}}

DEFINITIONS:
- knot_present:
  yes = at least one concrete local knot is visible in the text and remains unstable
        beyond a one-sentence self-check.
  no  = no concrete local knot; normal abstraction/exploration is allowed.

- knot_severity:
  0 = no knot
  1 = brief unstable span, then stable
  2 = moderate; active state becomes unreliable for a span of text
  3 = severe; prose becomes self-contradictory or cannot maintain a stable proof/search state

- knot_symptoms:
  choose up to 3 symptoms. Use ["none"] if no knot is present.

- primary_trigger:
  case_split          — active case/assumption tracking fails
  assumption_tracking — theorem precondition or temporary assumption drifts
  subgoal_frame       — local lemma/subgoal frame gets mixed with the main goal
  symbol_binding      — symbol/object identity or scope becomes unstable
  mixed               — more than one trigger is equally central
  unclear             — knot exists but trigger is unclear
  none                — no knot

- trace_strategy:
  structural_abstraction — abstract plan/proof outline with little local tracing
  symbolic_manipulation  — algebraic/symbolic derivation dominates
  case_analysis          — explicit case split reasoning dominates
  example_driven         — examples / concrete instantiations dominate
  patchy_backtracking    — repeated repairs / rewrites dominate

- state_consistency:
  stable             — state remains coherent
  minor_slip         — brief wobble, then stable
  lost_state         — trace loses which assumption/symbol/goal is active
  self_contradictory — directly inconsistent about active state
  not_applicable     — no real stateful proof/search attempt

FINAL CHECK BEFORE ANSWERING:
- If your quote does not itself display the knot, answer "no".
- If a careful human reader can still tell what assumption/symbol/goal is active, answer "no".
- Prefer false negatives over false positives.
"""


def build_prompt(record: dict, think_max_chars: int) -> str:
    problem_snippet = str(record.get("actual_prompt") or record.get("prompt") or "")[:500]
    return USER_TEMPLATE.format(
        problem_snippet=problem_snippet,
        think_text=record["think_text"],
        max_chars=think_max_chars,
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
        "max_tokens": 800,
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
                print("  Auth error — check GLM_API_KEY.")
                return {}
        except requests.Timeout:
            print(f"  Timeout (attempt {attempt}/{MAX_RETRIES})")
        except requests.RequestException as error:
            print(f"  Request error (attempt {attempt}/{MAX_RETRIES}): {error}")
        if attempt < MAX_RETRIES:
            print(f"  Waiting {delay}s ...")
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


def log_failure(failed_log: Path, record: dict, reason: str):
    entry = {
        "dataset": record["dataset"],
        "problem_id": record["problem_id"],
        "run_index": int(record["run_index"]),
        "reason": reason,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(failed_log, "a") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def log_failure_locked(failed_log: Path, lock: threading.Lock, record: dict, reason: str):
    with lock:
        log_failure(failed_log, record, reason)


def save_result(out_dir: Path, record: dict, glm_content: str):
    parsed, parse_error = parse_glm_content(glm_content)
    payload = {
        "dataset": record["dataset"],
        "problem_id": record["problem_id"],
        "problem_key": record["problem_key"],
        "run_index": int(record["run_index"]),
        "is_correct": int(record["is_correct"]),
        "source_report": record["source_report"],
        "think_chars": len(record["think_text"]),
        "think_total_chars": int(record.get("think_total_chars", len(record["think_text"]))),
        "prompt_version": PROMPT_VERSION,
        "glm_raw_content": glm_content,
        "glm_parsed": parsed,
        "parse_error": parse_error,
    }
    with open(out_path(out_dir, record), "w") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_manifest(out_dir: Path, payload: dict):
    with open(out_dir / "_manifest.json", "w") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def run_one_label(
    record: dict,
    api_key: str,
    out_dir: Path,
    rate_limiter: RateLimiter,
    failed_log: Path,
    failure_lock: threading.Lock,
    model: str,
    think_max_chars: int,
) -> dict:
    if already_done(out_dir, record):
        return {"status": "skip", "problem_key": record["problem_key"], "run_index": record["run_index"]}

    prompt_text = build_prompt(record, think_max_chars=think_max_chars)
    rate_limiter.wait()
    glm_raw = call_glm(prompt_text, api_key, model=model)
    glm_content = extract_content(glm_raw)
    if not glm_content:
        log_failure_locked(failed_log, failure_lock, record, "empty_response")
        return {"status": "fail", "problem_key": record["problem_key"], "run_index": record["run_index"]}

    save_result(out_dir, record, glm_content)
    parsed, parse_error = parse_glm_content(glm_content)
    if parse_error:
        return {"status": "parse_error", "problem_key": record["problem_key"], "run_index": record["run_index"]}
    return {
        "status": "ok",
        "problem_key": record["problem_key"],
        "run_index": record["run_index"],
        "severity": parsed.get("knot_severity", ""),
        "trigger": parsed.get("primary_trigger", ""),
    }


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


def main():
    parser = argparse.ArgumentParser(description="GLM math knot labeling")
    parser.add_argument("--datasets", nargs="+", default=list(MATH_REPORTS.keys()), choices=list(MATH_REPORTS.keys()))
    parser.add_argument("--n-problems", type=int, default=48, help="Problems to sample in sample mode")
    parser.add_argument("--runs-per-problem", type=int, default=4)
    parser.add_argument("--all-eligible", action="store_true", help="Label every run from problems with both correct and incorrect outputs")
    parser.add_argument("--max-runs", type=int, default=0, help="Optional cap for smoke tests")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0, help="0 -> choose conservative default from CPU load")
    parser.add_argument("--requests-per-second", type=float, default=2.0, help="Global request rate across workers")
    parser.add_argument("--model", type=str, default=GLM_MODEL)
    parser.add_argument("--think-max-chars", type=int, default=16000)
    parser.add_argument("--api-key", type=str, default="", help="Prefer GLM_API_KEY env var")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    api_key = get_api_key(args.api_key)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    failed_log = out_dir / "_failed_runs.jsonl"

    print("Loading math reports ...")
    records, problem_keys, labels = load_math_runs(args.datasets, think_max_chars=args.think_max_chars)
    print(f"  {len(records)} runs across {len(set(problem_keys))} problems from {len(args.datasets)} datasets")

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
    print("CPU / load check:")
    print(json.dumps(cpu_info, ensure_ascii=False))

    dataset_counts = defaultdict(int)
    for problem_key in selected_problem_keys:
        dataset_counts[problem_key.split(":", 1)[0]] += 1
    write_manifest(
        out_dir,
        {
            "mode": mode_name,
            "datasets": args.datasets,
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

    rate_limiter = RateLimiter(args.requests_per_second)
    failure_lock = threading.Lock()
    pending_records = [record for record in selected_records if not already_done(out_dir, record)]
    print(f"Pending after resume check: {len(pending_records)}")
    if not pending_records:
        consolidate_to_jsonl(out_dir)
        print("Nothing to do.")
        return

    counts = defaultdict(int)
    completed = 0
    with futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(
                run_one_label,
                record,
                api_key,
                out_dir,
                rate_limiter,
                failed_log,
                failure_lock,
                args.model,
                args.think_max_chars,
            ): record
            for record in pending_records
        }

        for future in futures.as_completed(future_map):
            result = future.result()
            counts[result["status"]] += 1
            completed += 1
            if completed % 25 == 0 or result["status"] in {"fail", "parse_error"}:
                print(
                    f"[{completed:5d}/{len(pending_records)}] "
                    f"ok={counts['ok']} skip={counts['skip']} "
                    f"parse={counts['parse_error']} fail={counts['fail']}"
                )

    retry_records: list[dict] = []
    if failed_log.exists():
        seen: set[tuple[str, str, int]] = set()
        for line in open(failed_log):
            try:
                entry = json.loads(line)
                key = (entry["dataset"], str(entry["problem_id"]), int(entry["run_index"]))
                if key in seen:
                    continue
                seen.add(key)
                for record in selected_records:
                    if (
                        record["dataset"] == key[0]
                        and str(record["problem_id"]) == key[1]
                        and int(record["run_index"]) == key[2]
                        and not already_done(out_dir, record)
                    ):
                        retry_records.append(record)
                        break
            except Exception:
                pass

    if retry_records:
        print(f"Retrying {len(retry_records)} failed items sequentially ...")
        pacer = AdaptivePacer(
            sleep_min=max(0.5, 1.0 / max(args.requests_per_second, 1e-6)),
            sleep_max=30.0,
        )
        for idx, record in enumerate(retry_records, 1):
            if already_done(out_dir, record):
                continue
            glm_raw = call_glm(build_prompt(record, args.think_max_chars), api_key, model=args.model)
            glm_content = extract_content(glm_raw)
            if glm_content:
                save_result(out_dir, record, glm_content)
                counts["retry_ok"] += 1
                pacer.record(True)
            else:
                counts["retry_fail"] += 1
                pacer.record(False)
            if idx % 25 == 0:
                print(f"  retry {idx}/{len(retry_records)} ok={counts['retry_ok']} fail={counts['retry_fail']}")
            pacer.sleep()

    consolidate_to_jsonl(out_dir)
    print(f"\nDone. Status counts: {dict(counts)}")
    print(f"Raw JSONs in: {out_dir}")
    print("Next step: python scripts/analyze_glm_math_knot.py")


if __name__ == "__main__":
    main()
