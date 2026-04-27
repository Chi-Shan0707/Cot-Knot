from __future__ import annotations

import concurrent.futures as futures
import json
import os
import re
import sys
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Callable

import numpy as np
import requests


REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_nad_root() -> Path:
    candidates = [
        REPO_ROOT.parent / "NAD_Next",
        REPO_ROOT.parent.parent / "NAD_Next",
        REPO_ROOT.parent.parent.parent / "NAD_Next",
        Path("/home/jovyan/work/NAD_Next"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[-1].resolve()


NAD_ROOT = resolve_nad_root()
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
    def __init__(self, window: int = 10, fail_thresh: float = 0.5, sleep_min: float = 0.5, sleep_max: float = 30.0):
        self.window = window
        self.fail_thresh = fail_thresh
        self.sleep_min = sleep_min
        self.sleep_max = sleep_max
        self._history: deque[bool] = deque(maxlen=window)
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


def load_runs_from_reports(report_paths: dict[str, Path], think_max_chars: int) -> tuple[list[dict], list[str], list[int], dict[str, str]]:
    records: list[dict] = []
    problem_keys: list[str] = []
    labels: list[int] = []
    problem_groups: dict[str, str] = {}

    for dataset, report_path in report_paths.items():
        report = json.loads(report_path.read_text())
        for problem in report["results"]:
            problem_id = str(problem["problem_id"])
            prompt = str(problem.get("prompt") or "")
            problem_key = f"{dataset}:{problem_id}"
            problem_groups[problem_key] = dataset
            for run in problem.get("runs", []):
                generated_text = str(run.get("generated_text") or "")
                think_excerpt, think_total_chars = build_trace_excerpt(generated_text, max_chars=think_max_chars)
                record = {
                    "dataset": dataset,
                    "problem_id": problem_id,
                    "problem_key": problem_key,
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
                problem_keys.append(problem_key)
                labels.append(record["is_correct"])
    return records, problem_keys, labels, problem_groups


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
    n_problems: int = 64,
    runs_per_problem: int = 4,
    seed: int = 42,
    problem_groups: dict[str, str] | None = None,
    balance_groups: bool = False,
) -> tuple[list[int], list[str]]:
    by_problem: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for rec_idx, (problem_key, label) in enumerate(zip(problem_keys, labels)):
        by_problem[problem_key].append((rec_idx, int(label)))

    eligible_by_group: dict[str, list[str]] = defaultdict(list)
    problem_acc: dict[str, float] = {}
    for problem_key, items in by_problem.items():
        if not any(label == 1 for _, label in items):
            continue
        if not any(label == 0 for _, label in items):
            continue
        group = problem_groups.get(problem_key, "all") if problem_groups else "all"
        eligible_by_group[group].append(problem_key)
        problem_acc[problem_key] = sum(label for _, label in items) / len(items)

    if not eligible_by_group:
        return [], []

    if balance_groups:
        active_groups = [group for group in sorted(eligible_by_group) if eligible_by_group[group]]
        base = n_problems // len(active_groups)
        remainder = n_problems % len(active_groups)
        selected_problem_keys: list[str] = []
        for idx, group in enumerate(active_groups):
            need = base + (1 if idx < remainder else 0)
            selected_problem_keys.extend(
                _stratified_problem_sample(problem_acc, eligible_by_group[group], need)
            )
    else:
        eligible = [problem_key for group in sorted(eligible_by_group) for problem_key in eligible_by_group[group]]
        selected_problem_keys = _stratified_problem_sample(problem_acc, eligible, n_problems)

    selected_problem_keys = selected_problem_keys[:n_problems]
    rng = np.random.default_rng(seed)
    selected_indices: list[int] = []
    half = max(1, runs_per_problem // 2)

    for problem_key in selected_problem_keys:
        items = by_problem[problem_key]
        correct_idx = [rec_idx for rec_idx, label in items if label == 1]
        incorrect_idx = [rec_idx for rec_idx, label in items if label == 0]
        rng.shuffle(correct_idx)
        rng.shuffle(incorrect_idx)
        chosen = correct_idx[:half] + incorrect_idx[:half]
        if len(chosen) < runs_per_problem:
            all_idx = correct_idx + incorrect_idx
            rng.shuffle(all_idx)
            for rec_idx in all_idx:
                if rec_idx not in chosen:
                    chosen.append(rec_idx)
                if len(chosen) >= runs_per_problem:
                    break
        selected_indices.extend(chosen[:runs_per_problem])

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


def call_glm(system_prompt: str, prompt_text: str, api_key: str, model: str = GLM_MODEL) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
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


def save_result(out_dir: Path, domain: str, prompt_version: str, record: dict, glm_content: str):
    parsed, parse_error = parse_glm_content(glm_content)
    payload = {
        "domain": domain,
        "dataset": record["dataset"],
        "problem_id": record["problem_id"],
        "problem_key": record["problem_key"],
        "run_index": int(record["run_index"]),
        "is_correct": int(record["is_correct"]),
        "source_report": record["source_report"],
        "think_chars": len(record["think_text"]),
        "think_total_chars": int(record.get("think_total_chars", len(record["think_text"]))),
        "prompt_version": prompt_version,
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
    domain: str,
    prompt_version: str,
    api_key: str,
    system_prompt: str,
    prompt_builder: Callable[[dict, int], str],
    out_dir: Path,
    rate_limiter: RateLimiter,
    failed_log: Path,
    failure_lock: threading.Lock,
    model: str,
    think_max_chars: int,
) -> dict:
    if already_done(out_dir, record):
        return {"status": "skip", "problem_key": record["problem_key"], "run_index": record["run_index"]}

    prompt_text = prompt_builder(record, think_max_chars)
    rate_limiter.wait()
    glm_raw = call_glm(system_prompt, prompt_text, api_key, model=model)
    glm_content = extract_content(glm_raw)
    if not glm_content:
        log_failure_locked(failed_log, failure_lock, record, "empty_response")
        return {"status": "fail", "problem_key": record["problem_key"], "run_index": record["run_index"]}

    save_result(out_dir, domain, prompt_version, record, glm_content)
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


def execute_labeling(
    *,
    selected_records: list[dict],
    domain: str,
    prompt_version: str,
    api_key: str,
    system_prompt: str,
    prompt_builder: Callable[[dict, int], str],
    out_dir: Path,
    workers: int,
    requests_per_second: float,
    model: str,
    think_max_chars: int,
    progress_every: int = 25,
) -> dict[str, int]:
    failed_log = out_dir / "_failed_runs.jsonl"
    rate_limiter = RateLimiter(requests_per_second)
    failure_lock = threading.Lock()
    pending_records = [record for record in selected_records if not already_done(out_dir, record)]
    print(f"Pending after resume check: {len(pending_records)}")
    if not pending_records:
        consolidate_to_jsonl(out_dir)
        return {"skip": len(selected_records)}

    counts: dict[str, int] = defaultdict(int)
    completed = 0
    with futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(
                run_one_label,
                record,
                domain,
                prompt_version,
                api_key,
                system_prompt,
                prompt_builder,
                out_dir,
                rate_limiter,
                failed_log,
                failure_lock,
                model,
                think_max_chars,
            ): record
            for record in pending_records
        }

        for future in futures.as_completed(future_map):
            result = future.result()
            counts[result["status"]] += 1
            completed += 1
            if completed % progress_every == 0 or result["status"] in {"fail", "parse_error"}:
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
            sleep_min=max(0.5, 1.0 / max(requests_per_second, 1e-6)),
            sleep_max=30.0,
        )
        for idx, record in enumerate(retry_records, 1):
            if already_done(out_dir, record):
                continue
            glm_raw = call_glm(system_prompt, prompt_builder(record, think_max_chars), api_key, model=model)
            glm_content = extract_content(glm_raw)
            if glm_content:
                save_result(out_dir, domain, prompt_version, record, glm_content)
                counts["retry_ok"] += 1
                pacer.record(True)
            else:
                counts["retry_fail"] += 1
                pacer.record(False)
            if idx % progress_every == 0:
                print(f"  retry {idx}/{len(retry_records)} ok={counts['retry_ok']} fail={counts['retry_fail']}")
            pacer.sleep()

    consolidate_to_jsonl(out_dir)
    return dict(counts)
