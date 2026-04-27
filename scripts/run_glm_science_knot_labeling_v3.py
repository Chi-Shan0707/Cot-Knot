"""
GLM Science Knot Labeling v3
============================
ALIGNED with math protocol: uses the exact same 6 symptom names as the math
knot labeling script (run_glm_math_knot_labeling.py), adapted for science.

v1 failed: 100% positive (added assumption_contradiction — not in math schema)
v2 failed: 9% positive, wrong direction (different symptom taxonomy)
v3: same 6 math symptoms, science-adapted examples, 240-run target

Symptoms (identical names to math):
  case_split_instability  — considers case/mechanism A, silently reasons with B
  assumption_drift        — explicit regime/precondition stated, then silently violated
  subgoal_frame_loss      — sub-calculation context mixed with main goal
  variable_binding_drift  — symbol used with different physical identity
  invariant_drift         — explicit conservation law invoked, then violated
  repair_without_recovery — ≥3 repair attempts, same claim, still unstable

Key fix from v1: NO assumption_contradiction (that was extra and too broad).
Key fix from v1: repair_without_recovery requires ≥3 attempts AND persistent instability.

Outputs:
  results/glm_science_knot_raw_v3/

Usage:
  GLM_API_KEY=<key> python scripts/run_glm_science_knot_labeling_v3.py --max-runs 10
  GLM_API_KEY=<key> python scripts/run_glm_science_knot_labeling_v3.py --n-problems 60
  GLM_API_KEY=<key> python scripts/run_glm_science_knot_labeling_v3.py --all-eligible
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
PROMPT_VERSION = "v3"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "glm_science_knot_raw_v3"

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
            print(f"\n  High failure rate ({fail_rate:.0%}). Slowing to {self._current_sleep:.1f}s.")
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
        for run in problem.get("runs", []):
            generated_text = str(run.get("generated_text") or "")
            think_excerpt, think_total_chars = build_trace_excerpt(generated_text, max_chars=think_max_chars)
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
    n_problems: int = 60,
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
    "You are an expert at reading graduate-level science chain-of-thought traces "
    "(physics, chemistry, biology). Your task is to detect local reasoning-knot points: "
    "places where the prose loses stable control of the active mechanism, physical regime, "
    "symbol, subgoal, or tracked conservation law. "
    "Do NOT score generic hesitation, exploration of hypotheses, or overall correctness. "
    "Respond in valid JSON only."
)


USER_TEMPLATE = """\
A model is answering a graduate-level science multiple-choice question (GPQA).
Read its chain-of-thought reasoning carefully.

[PROBLEM SNIPPET]
{problem_snippet}

[MODEL REASONING EXCERPTS (up to {max_chars} chars total)]
{think_text}

---
BACKGROUND — what to detect
---
The displayed reasoning may be non-contiguous slices. Only use what is visible.
Do not infer a knot from omitted regions or truncation.

We are NOT scoring correctness, numerical errors, or generic uncertainty.
We look for LOCAL REASONING KNOTS: places where the text loses a stable scientific
reasoning state. A knot requires a concrete, visible instability in the PROSE ITSELF.

CONSERVATIVE LABELING RULE:
- Only answer knot_present="yes" if you can justify it with a verbatim quote.
- If reasoning is exploratory but scientifically coherent, answer "no".
- A single clean self-correction that restores stable state is healthy — answer "no".
- Being wrong is not enough. Local state instability in the prose is required.
- If the quote shows only variable introduction, equation setup, or hypothesis
  exploration, answer "no".
- If instability resolves within the next two sentences and state is readable, answer "no".
- To answer "yes", the excerpt must show one of these concrete forms:
  (a) explicit contradiction between two simultaneous active claims,
  (b) incompatible redefinition of the same physical quantity or symbol,
  (c) mechanism/case switch without closing the prior thread, or
  (d) ≥ 3 repair attempts on the SAME claim with no convergence.

---
SCIENCE-SPECIFIC NEGATIVE EXAMPLES — always answer "no" for these:
---

1. Reconsidering a mechanism (NOT a knot):
   "Let me first check if this is SN1... but the substrate is primary, so SN2 is
   more likely. Therefore inversion gives the (R) product."
   Reason: clean single-mechanism switch with closure. State is stable.

2. Normal "wait/actually" (NOT a knot):
   "ΔG = ΔH - TΔS = 30 - 298×0.08 = 30 - 23.8 = 6.2 kJ/mol. Wait, let me
   recalculate: 298 × 0.08 = 23.84, so ΔG ≈ 6.2 kJ/mol. Confirmed."
   Reason: one arithmetic check, same answer. Healthy. NOT a knot.

3. Exploring multiple hypotheses in sequence (NOT a knot):
   "Could be aldehyde (1720 cm⁻¹) or ketone (1715 cm⁻¹). Given no α-H, ketone."
   Reason: orderly hypothesis elimination. NOT a knot.

4. Considering approximation and then using it (NOT a knot):
   "Assuming ideal gas: PV = nRT. Then P = nRT/V = ..."
   Reason: the assumption is stated and then correctly used. NOT a knot.

5. Formula application with unit checking (NOT a knot):
   "E = hν = 6.63×10⁻³⁴ × 5×10¹⁴ = 3.3×10⁻¹⁹ J."
   Reason: direct formula application. NOT a knot even if the number is wrong.

---
CORE SCIENCE KNOT SYMPTOMS (same structure as math protocol):
---

1. case_split_instability
   The trace enters a physical case or mechanism (e.g., "assuming pathway A"),
   then silently reasons with properties specific to a DIFFERENT case/mechanism
   while still nominally within the first, without a clean handoff.
   In science: e.g., opens "Case 1: quantum tunneling dominates," then uses the
   classical thermal activation formula for that case's calculation.
   DO NOT label: clean sequential consideration of multiple mechanisms.

2. assumption_drift
   An approximation, physical regime, or formula precondition is EXPLICITLY STATED
   first, then the trace uses a formula that directly CONTRADICTS that stated
   condition — without acknowledging the contradiction.
   In science: e.g., "We apply the weak-field approximation (μB ≪ kT)" → then
   uses the strong-field Zeeman formula (with separate m_J contributions) for the
   same calculation, without noting the regime change.
   DO NOT label: simply using a formula without first stating its conditions.
   DO NOT label: reconsidering which approximation is appropriate.

3. subgoal_frame_loss
   The trace sets up a specific sub-calculation ("first find the activation energy
   for step 2"), then mixes up the sub-result with the main quantity, or applies
   the sub-result in the wrong context.
   In science: e.g., sets up to find "the rate constant k₂", then substitutes
   k₂'s formula into the expression for k₁ without noting the swap.
   DO NOT label: cleanly completing a sub-calculation and using the result.

4. variable_binding_drift
   A physical symbol, chemical species label, or quantity introduced for one
   purpose is later used with a DIFFERENT physical identity in the same calculation.
   In science: e.g., "let E = electric field = 10⁵ V/m" ... then "E = ½mv² = ..."
   where E now means kinetic energy — both in the same running calculation.
   DO NOT label: reusing symbol names in clearly separate contexts or equations.
   DO NOT label: standard substitution (replacing known values into formulas).

5. invariant_drift
   The trace explicitly states a conservation law or physical constraint ("momentum
   is conserved," "total charge is conserved"), then performs a calculation that
   violates that explicitly stated conservation law without acknowledging it.
   In science: e.g., "by conservation of energy: E_before = E_after" → then
   adds an energy term on one side but not the other without justification.
   DO NOT label: not mentioning a conservation law at all.
   DO NOT label: approximately conserved quantities with acknowledged corrections.

6. repair_without_recovery
   ≥ 3 explicit repair phrases ("wait / actually / no / but") applied to the SAME
   specific scientific claim, with each attempt yielding a DIFFERENT contradictory
   answer, and the state REMAINING UNSTABLE (no convergence) after the last repair.
   In science: e.g., three consecutive "wait, the product is X / no wait, it's Y /
   actually it should be Z" about the SAME reaction product, with no final stable claim.
   DO NOT label: 1-2 repairs that converge to a stable final answer.
   DO NOT label: exploring different sub-problems in sequence.
   DO NOT label: ≥3 repairs that ultimately converge to one stable answer.

---
POSITIVE EXAMPLES (answer "yes"):
---

Example 1 (assumption_drift):
"Under the Born-Oppenheimer approximation, nuclear and electronic motion separate.
So we can treat the nuclei as fixed. [20 lines of other reasoning]
Now accounting for nuclear kinetic energy in the electronic Hamiltonian, we get..."
Reason: BO approximation was explicitly stated (treat nuclei as fixed), then the
trace includes nuclear kinetic energy in the Hamiltonian — directly contradicting
the stated BO approximation — without noting any departure from it.

Example 2 (variable_binding_drift):
"Define Δ = orbital splitting energy = 20,000 cm⁻¹ for this octahedral complex.
[5 lines] Therefore the bond angle Δ determines the geometry as tetrahedral."
Reason: Δ was defined as orbital splitting (energy units) but is then used as a
geometric bond angle concept. Same symbol, two incompatible physical identities,
active simultaneously without reassignment.

---
TASK
---
Respond with a JSON object with EXACTLY these 9 keys:

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
- knot_present: yes = at least one concrete local knot visible in text, persisting beyond one-sentence self-check. no = no concrete local knot.
- knot_severity: 0=none, 1=brief then stable, 2=active state unreliable for a span, 3=self-contradictory/unresolvable.
- knot_symptoms: up to 3. Use ["none"] if no knot.
- primary_trigger: case_split / assumption_tracking / subgoal_frame / symbol_binding / mixed / unclear / none.
- trace_strategy: structural_abstraction / symbolic_manipulation / case_analysis / example_driven / patchy_backtracking.
- state_consistency: stable / minor_slip / lost_state / self_contradictory / not_applicable.

FINAL CHECK:
- Quote must itself display the knot. If not, answer "no".
- If a careful reader can still track which mechanism/formula/symbol is active, answer "no".
- Strongly prefer false negatives over false positives.
"""


def build_prompt(record: dict, think_max_chars: int) -> str:
    problem_snippet = str(record.get("actual_prompt") or record.get("prompt") or "")[:500]
    return USER_TEMPLATE.format(
        problem_snippet=problem_snippet,
        think_text=record["think_text"],
        max_chars=think_max_chars,
    )


def call_glm(prompt_text: str, api_key: str, model: str) -> dict:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
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
    entry = {"dataset": record["dataset"], "problem_id": record["problem_id"],
             "run_index": int(record["run_index"]), "reason": reason,
             "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with lock:
        with open(failed_log, "a") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def save_result(out_dir: Path, record: dict, glm_content: str):
    parsed, parse_error = parse_glm_content(glm_content)
    payload = {
        "dataset": record["dataset"], "problem_id": record["problem_id"],
        "problem_key": record["problem_key"], "run_index": int(record["run_index"]),
        "is_correct": int(record["is_correct"]), "source_report": record["source_report"],
        "think_chars": len(record["think_text"]),
        "think_total_chars": int(record.get("think_total_chars", len(record["think_text"]))),
        "prompt_version": PROMPT_VERSION,
        "glm_raw_content": glm_content, "glm_parsed": parsed, "parse_error": parse_error,
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


def run_one_label(record, api_key, out_dir, rate_limiter, failed_log, failure_lock, model, think_max_chars):
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
    return {"status": "ok", "problem_key": record["problem_key"], "run_index": record["run_index"],
            "knot_present": parsed.get("knot_present", ""), "severity": parsed.get("knot_severity", "")}


def main():
    parser = argparse.ArgumentParser(description="GLM science knot labeling v3 (math-aligned)")
    parser.add_argument("--n-problems", type=int, default=60, help="default 60 → 240 runs at runs-per-problem=4")
    parser.add_argument("--runs-per-problem", type=int, default=4)
    parser.add_argument("--all-eligible", action="store_true")
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--requests-per-second", type=float, default=3.0)
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
            n_problems=args.n_problems, runs_per_problem=args.runs_per_problem, seed=args.seed,
        )
        mode_name = "sample"

    if args.max_runs > 0:
        selected_indices = selected_indices[: args.max_runs]

    selected_records = [records[idx] for idx in selected_indices]
    print(f"Selected {len(selected_records)} runs across {len(selected_problem_keys)} problems ({mode_name})")

    workers, cpu_info = recommend_workers()
    if args.workers > 0:
        workers = args.workers
    print(f"Workers: {workers}")

    write_manifest(out_dir, {
        "mode": mode_name, "dataset": "gpqa", "protocol": "v3_math_aligned",
        "selected_runs": len(selected_records), "selected_problems": len(selected_problem_keys),
        "workers": workers, "requests_per_second": args.requests_per_second,
        "model": args.model, "think_max_chars": args.think_max_chars,
        "prompt_version": PROMPT_VERSION, "seed": args.seed,
        "cpu_info": cpu_info, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })

    rate_limiter = RateLimiter(args.requests_per_second)
    failure_lock = threading.Lock()
    pending_records = [record for record in selected_records if not already_done(out_dir, record)]
    print(f"Pending: {len(pending_records)}")
    if not pending_records:
        consolidate_to_jsonl(out_dir)
        print("Nothing to do.")
        return

    counts = defaultdict(int)
    completed = 0
    with futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(run_one_label, record, api_key, out_dir,
                            rate_limiter, failed_log, failure_lock,
                            args.model, args.think_max_chars): record
            for record in pending_records
        }
        for future in futures.as_completed(future_map):
            result = future.result()
            counts[result["status"]] += 1
            completed += 1
            if completed % 50 == 0 or result["status"] in {"fail", "parse_error"}:
                print(f"[{completed:5d}/{len(pending_records)}] "
                      f"ok={counts['ok']} skip={counts['skip']} "
                      f"parse={counts['parse_error']} fail={counts['fail']}")

    # Retry
    retry_records: list[dict] = []
    if failed_log.exists():
        seen: set = set()
        for line in open(failed_log):
            try:
                entry = json.loads(line)
                key = (entry["dataset"], str(entry["problem_id"]), int(entry["run_index"]))
                if key in seen:
                    continue
                seen.add(key)
                for record in selected_records:
                    if (record["dataset"] == key[0] and str(record["problem_id"]) == key[1]
                            and int(record["run_index"]) == key[2] and not already_done(out_dir, record)):
                        retry_records.append(record)
                        break
            except Exception:
                pass

    if retry_records:
        print(f"Retrying {len(retry_records)} failed items ...")
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
    print(f"Output: {out_dir}")
    print("Next step: python scripts/analyze_glm_science_knot_v3.py")


if __name__ == "__main__":
    main()
