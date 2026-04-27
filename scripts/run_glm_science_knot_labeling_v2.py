"""
GLM Science Knot Labeling v2 — Replacement Protocol
=====================================================
This is the REPLACEMENT protocol for science (GPQA) knot annotation.
v1 produced 100% positive labels (degenerate collapse, analogous to the original
coding GLM protocol that yielded κ = 0.0).

v2 tightens the definition to focus on EXPLICIT, UNRESOLVED scientific
contradictions — analogous to the coding "execution-semantic break" replacement.

Core change from v1:
  - Remove assumption_contradiction (too broad; catches normal exploration)
  - Remove variable_identity_drift (too broad)
  - Introduce explicit_claim_reversal: trace makes claim X then claim ¬X about
    the SAME entity, WITHOUT reconciliation
  - Introduce irreparable_regression: ≥4 repair cycles on the same sub-problem
    yielding contradictory claims, never reaching stable state
  - Keep framework_collision (≡ concept_conflation v1) and formula_scope_violation
  - Require all instabilities to PERSIST > 3 sentences
  - Add many negative examples for normal science exploration

Outputs:
  results/glm_science_knot_raw_v2/
    - one JSON per labeled run
    - all_runs.jsonl
    - _manifest.json
    - _failed_runs.jsonl

Usage:
  GLM_API_KEY=<key> python scripts/run_glm_science_knot_labeling_v2.py --max-runs 10
  GLM_API_KEY=<key> python scripts/run_glm_science_knot_labeling_v2.py --n-problems 30
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
_candidate = REPO_ROOT.parent / "NAD_Next"
NAD_ROOT = _candidate if _candidate.exists() else REPO_ROOT.parent.parent.parent / "NAD_Next"
PROMPT_VERSION = "v2"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "glm_science_knot_raw_v2"

SCIENCE_REPORT = NAD_ROOT / "MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/gpqa/cache_neuron_output_1_act_no_rms_20251126_111853/evaluation_report.json"

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


def load_science_runs(think_max_chars: int) -> tuple[list[dict], list[str], list[int]]:
    records: list[dict] = []
    problem_keys: list[str] = []
    labels: list[int] = []

    report = json.loads(SCIENCE_REPORT.read_text())
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
                "dataset": "gpqa",
                "problem_id": problem_id,
                "problem_key": f"gpqa:{problem_id}",
                "run_index": int(run["run_index"]),
                "is_correct": int(bool(run.get("is_correct", False))),
                "prompt": prompt,
                "actual_prompt": str(run.get("actual_prompt") or ""),
                "generated_text": generated_text,
                "think_text": think_excerpt,
                "think_total_chars": think_total_chars,
                "source_report": str(SCIENCE_REPORT),
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
    n_problems: int = 30,
    runs_per_problem: int = 4,
    seed: int = 42,
) -> tuple[list[int], list[str]]:
    by_problem: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for rec_idx, (problem_key, label) in enumerate(zip(problem_keys, labels)):
        by_problem[problem_key].append((rec_idx, int(label)))

    eligible: list[str] = []
    problem_acc: dict[str, float] = {}
    for problem_key, items in by_problem.items():
        if not any(label == 1 for _, label in items):
            continue
        if not any(label == 0 for _, label in items):
            continue
        eligible.append(problem_key)
        problem_acc[problem_key] = sum(label for _, label in items) / len(items)

    selected_problem_keys = _stratified_problem_sample(problem_acc, eligible, n_problems)

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
    "You are an expert at reading graduate-level science chain-of-thought traces. "
    "Your task is to identify EXPLICIT, UNRESOLVED scientific contradictions — places "
    "where the trace commits to two incompatible scientific claims about the same "
    "entity WITHOUT reconciling them. Normal exploration, hypothesis testing, and "
    "self-correction ARE NOT contradictions. Be very conservative. "
    "Respond in valid JSON only."
)


USER_TEMPLATE = """\
A model is answering a graduate-level science multiple-choice question.
Read its chain-of-thought reasoning.

[PROBLEM SNIPPET]
{problem_snippet}

[MODEL REASONING EXCERPTS (up to {max_chars} chars total)]
{think_text}

---
WHAT TO DETECT — very narrow definition
---
This is a REPLACEMENT protocol for science knot annotation.
The previous protocol failed by marking 100% of traces as positive.
This protocol is deliberately much more conservative.

We are looking for only ONE thing: an EXPLICIT, UNRESOLVED scientific contradiction.
This means the trace makes a concrete factual claim about a specific entity or
quantity, and then makes a different, incompatible claim about the SAME entity or
quantity, without resolving the contradiction in the next 3 sentences.

---
WHAT DOES NOT COUNT AS A KNOT (do not mark these):
---
- Reconsidering a hypothesis: "Let me try X... wait, let me reconsider and try Y instead."
  → This is normal exploration. NOT a knot.
- Normal backtracking: "Actually, I made an error. Let me recalculate..."
  → If the recalculation reaches a stable result, NOT a knot.
- Uncertainty about which answer to choose: "It could be A or B..."
  → Weighing options is normal. NOT a knot.
- Using "wait", "actually", "no", "but" 1-3 times: normal self-correction. NOT a knot.
- Considering multiple mechanisms before settling on one: NOT a knot.
- Arriving at a wrong numerical answer (unless the trace ALSO claims contradictory
  values for the same quantity simultaneously): NOT a knot.
- Imprecise language or approximations: NOT a knot.
- Incomplete reasoning at the end of a trace slice: NOT a knot.
- Changing which answer option (A/B/C/D) is preferred: NOT a knot.
- Long, winding reasoning with many revisions: NOT a knot by itself.

---
WHAT DOES COUNT AS A KNOT (mark these):
---
The bar is high. Only mark "yes" if ALL of these are true:

1. The trace makes a CONCRETE, EXPLICIT claim: "X = [value]" or "substance X is [property]"
   or "the mechanism is [specific mechanism]" or "this is [formula]".
2. LATER in the SAME trace, without a reconciliation, the trace makes an
   INCOMPATIBLE claim about the SAME object X: "X = [different value]" or
   "substance X is [contradictory property]".
3. The contradiction PERSISTS for more than 3 sentences — it is not immediately
   corrected or explained.
4. You can provide a VERBATIM QUOTE showing both the original claim and the
   contradictory claim, both visible in the shown text.

---
SYMPTOM TAXONOMY (v2 — narrowed):
---

1. explicit_claim_reversal
   The trace explicitly states fact F about entity X (e.g., "the product has formula
   C8H10"), then later explicitly states the contradictory ¬F about the same X
   (e.g., "the product has formula C9H12") without explanation, and the contradiction
   persists for >3 sentences.
   DOES NOT INCLUDE: changing one's mind after a calculation error is caught.

2. framework_collision
   The trace simultaneously applies two mutually EXCLUSIVE physical or chemical
   frameworks to the EXACT SAME entity in a way that produces directly contradictory
   predictions — and does not separate these into "let's check if framework A applies,
   then check if framework B applies."
   DOES NOT INCLUDE: comparing quantum vs classical predictions in sequence.

3. formula_scope_violation
   The trace applies a formula that EXPLICITLY requires condition C, but the trace
   has ALSO explicitly stated that condition C does not hold — and then uses the
   formula's result without acknowledging the violation.
   DOES NOT INCLUDE: using standard approximations without stating their limits.

4. irreparable_regression
   The trace makes ≥ 4 explicit repair attempts ("wait / actually / no / but") on
   the SAME specific sub-claim (same quantity, same entity), with EACH repair
   producing a DIFFERENT contradictory answer, and the trace NEVER converges to a
   stable claim for that sub-problem.
   DOES NOT INCLUDE: multiple corrections that converge to a stable final answer.

---
NEGATIVE EXAMPLES (all answer "no"):
---

Example 1 — Reconsidering hypothesis (NOT a knot):
"Let me try option B, which would give ΔG = -20 kJ/mol. But wait, that assumes
standard conditions. Let me reconsider: the problem says T = 400K, so I should
use ΔG = ΔH - TΔS = 50 - 400×0.15 = -10 kJ/mol. So the answer is C."
Reason: the model reconsidered and reached a stable conclusion. NOT a knot.

Example 2 — Normal "wait/actually" (NOT a knot):
"The reaction is SN2, so inversion occurs. Wait, actually the substrate is tertiary,
so SN1 is more likely and racemization occurs instead."
Reason: one clean self-correction that reaches stable state. NOT a knot.

Example 3 — Multiple repairs converging (NOT a knot):
"Molecular formula: 12+14+16=42. No wait: C3H6O = 3×12 + 6×1 + 16 = 58.
Actually that's wrong too: 36+6+16=58. Yes, 58."
Reason: multiple arithmetic corrections, but they converge to 58. NOT a knot.

Example 4 — Considering multiple options (NOT a knot):
"It might be an aldehyde (C=O at 1720) or a ketone (C=O at 1715). Given that
there's no α-H signal, it's likely a ketone."
Reason: normal process of elimination. NOT a knot.

Example 5 — Long winding trace (NOT a knot):
Even if the trace is very long and winding with many "wait" words, if a careful
reader can still identify what the trace currently believes is true, it is NOT a knot.

---
POSITIVE EXAMPLES (answer "yes"):
---

Example P1 — explicit_claim_reversal:
"...so the ester has formula C4H8O2. [20 lines of other reasoning] ...
therefore the product C5H10O2 reacts with the Grignard reagent to give C4H8O2..."
[The trace now claims C4H8O2 is the Grignard product, but earlier said C4H8O2 was
the starting ester. Same formula, two different entities. Neither claim is retracted.]
Reason: two incompatible assignments of the same formula to different roles,
both active simultaneously. Quote would include both sentences.

Example P2 — framework_collision:
"This is a quantum system, so we must use Ψ = Ae^(ikx). The electron energy is
E = p²/2m = (ħk)²/2m. Therefore for this quantum electron, the classical
orbit radius is r = mv²/eE giving Bohr radius 0.529 Å."
Reason: the trace uses quantum wave mechanics AND then directly applies a classical
orbit equation (r = mv²/eE) to the SAME quantum electron, with both applied
simultaneously without transition. This is a direct framework collision.

---
TASK
---
Respond with a JSON object containing EXACTLY these 9 keys:

{{
  "knot_present": "<yes|no>",
  "knot_severity": <0|1|2|3>,
  "knot_symptoms": ["<choose 0-3 from: explicit_claim_reversal, framework_collision, formula_scope_violation, irreparable_regression, none>"],
  "primary_trigger": "<claim_reversal|framework_boundary|formula_precondition|regression_loop|mixed|unclear|none>",
  "knot_quote": "<verbatim ≤30 words from the reasoning showing the contradiction, or empty string>",
  "trace_strategy": "<principle_application|formula_derivation|case_elimination|mechanism_tracing|patchy_backtracking>",
  "reversal_count": <integer, count of explicit repair turns like wait/actually/no/but>,
  "state_consistency": "<stable|minor_slip|lost_state|self_contradictory|not_applicable>",
  "open_diagnosis": "<1-2 sentence concrete diagnosis of the contradiction, or why no contradiction exists>"
}}

DEFINITIONS:
- knot_present:
  yes = the trace contains an EXPLICIT, UNRESOLVED, PERSISTENT (>3 sentences)
        contradiction between two concrete claims about the same entity.
  no  = no such explicit unresolved contradiction; exploration and self-correction
        are not contradictions.

- knot_severity:
  0 = no knot
  1 = mild: two conflicting claims visible but trace remains mostly followable
  2 = moderate: the contradiction creates sustained confusion about the claim
  3 = severe: the contradiction is central and the trace never resolves it

- state_consistency:
  stable             — no unresolved contradictions between concrete claims
  minor_slip         — brief inconsistency, immediately self-corrected
  lost_state         — multiple contradictory active claims, unresolved
  self_contradictory — explicit contradiction between two simultaneous claims
  not_applicable     — no concrete claims made (pure exploration)

FINAL CHECK:
- Would a PhD scientist reading this trace be confused about what the trace
  currently claims to be true? If yes → potentially a knot.
  If no → NOT a knot. This is the primary test.
- Prefer false negatives strongly over false positives.
- If in doubt, answer "no".
"""


def build_prompt(record: dict, think_max_chars: int) -> str:
    problem_snippet = str(record.get("actual_prompt") or record.get("prompt") or "")[:600]
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


def log_failure_locked(failed_log: Path, lock: threading.Lock, record: dict, reason: str):
    entry = {
        "dataset": record["dataset"],
        "problem_id": record["problem_id"],
        "run_index": int(record["run_index"]),
        "reason": reason,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with lock:
        with open(failed_log, "a") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


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
        "knot_present": parsed.get("knot_present", ""),
        "severity": parsed.get("knot_severity", ""),
    }


def main():
    parser = argparse.ArgumentParser(description="GLM science (GPQA) knot labeling v2 — replacement protocol")
    parser.add_argument("--n-problems", type=int, default=30)
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
    failed_log = out_dir / "_failed_runs.jsonl"

    print("Loading GPQA science report ...")
    records, problem_keys, labels = load_science_runs(think_max_chars=args.think_max_chars)
    print(f"  {len(records)} runs across {len(set(problem_keys))} problems")

    if args.all_eligible:
        selected_indices, selected_problem_keys = all_eligible_runs(problem_keys, labels, seed=args.seed)
        mode_name = "all_eligible"
    else:
        selected_indices, selected_problem_keys = sample_runs(
            records, problem_keys, labels,
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

    write_manifest(out_dir, {
        "mode": mode_name,
        "dataset": "gpqa",
        "protocol": "v2_replacement",
        "selected_runs": len(selected_records),
        "selected_problems": len(selected_problem_keys),
        "workers": workers,
        "requests_per_second": args.requests_per_second,
        "model": args.model,
        "think_max_chars": args.think_max_chars,
        "prompt_version": PROMPT_VERSION,
        "seed": args.seed,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })

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
                run_one_label, record, api_key, out_dir,
                rate_limiter, failed_log, failure_lock,
                args.model, args.think_max_chars,
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

    # Sequential retry
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
        pacer = AdaptivePacer(sleep_min=max(0.5, 1.0 / max(args.requests_per_second, 1e-6)), sleep_max=30.0)
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
            pacer.sleep()

    consolidate_to_jsonl(out_dir)
    print(f"\nDone. Status counts: {dict(counts)}")
    print(f"Raw JSONs in: {out_dir}")
    print("Next step: python scripts/analyze_glm_science_knot_v2.py")


if __name__ == "__main__":
    main()
