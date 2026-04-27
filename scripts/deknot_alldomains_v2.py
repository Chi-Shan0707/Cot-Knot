#!/usr/bin/env python3
"""
De-knotting experiment v2 — all three domains.

Key fix vs v1 (deknot_coding_experiment.py):
  v1 only spliced the decoded text string.
  tok_conf_prefix was still mean(full conf array) — knot tokens not removed.

v2 properly masks ALL five per-token arrays (tok_conf, tok_gini, tok_logprob,
tok_neg_entropy, tok_selfcert) by mapping GLM char spans → token spans via the
fast tokenizer's offset_mapping, then computes each per-token feature as the
mean over non-knot tokens only.

Domains: math (aime24), science (gpqa), coding (livecodebench_v5).

Usage:
    cd /home/jovyan/work/NAD_Next && source .venv/bin/activate
    python /home/jovyan/work/SVDomain/workshop/cotknot/scripts/deknot_alldomains_v2.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from openai import OpenAI
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoTokenizer

# ── Constants ─────────────────────────────────────────────────────────────────

CACHE_BASE = Path("/home/jovyan/public-ro/MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B")
TOKENIZER_PATH = "/home/jovyan/public-ro/model/DeepSeek-R1-0528-Qwen3-8B"
GLM_KEY   = "49bbfdfe02384b74b6882da123bd6ee6.Xp4qZQTqnS3s5130"
GLM_BASE  = "https://open.bigmodel.cn/api/paas/v4/"
GLM_MODEL = "glm-4-flash"
OUT_DIR   = Path("/home/jovyan/work/SVDomain/workshop/cotknot/results/tables")

N_SAMPLE     = 200   # 100 correct + 100 incorrect per domain
SEED         = 42
SLICE_WORDS  = 40
REFL_THRESH  = 0.30
GLM_WORKERS  = 8
GLM_RETRIES  = 2
TEXT_WINDOW  = 4000  # chars sent to GLM for knot detection

# ── Domain configs ─────────────────────────────────────────────────────────────

DOMAINS = {
    "math": {
        "cache_path": CACHE_BASE / "aime24" / "cache_neuron_output_1_act_no_rms_20251126_073502",
    },
    "science": {
        "cache_path": CACHE_BASE / "gpqa" / "cache_neuron_output_1_act_no_rms_20251126_111853",
    },
    "coding": {
        "cache_path": CACHE_BASE / "livecodebench_v5" / "cache_neuron_output_1_act_no_rms_20251127_032808",
    },
}

# ── Domain-specific GLM prompts ───────────────────────────────────────────────

_PROMPT_MATH = """\
You are analyzing a chain-of-thought (CoT) reasoning trace from an LLM solving a math competition problem.

Identify "knot" segments: places where the reasoning goes in circles without making progress.
Specifically look for:
- Repeated re-derivation of the same intermediate result
- Incorrect backtracking that undoes correct progress and revisits the same dead end
- Extended metacognitive loops ("let me check... actually wait... let me recheck...") with no new content
- Repeated failed algebraic manipulations of the same expression

Return ONLY valid JSON:
{{"knots": [{{"start": <int>, "end": <int>}}, ...]}}

Rules:
- start/end are 0-indexed character positions in the text
- Each span must be >= 50 chars; spans must not overlap
- Return {{"knots": []}} if no knots found

Analyze only the first {window} characters:
{text_slice}
"""

_PROMPT_SCIENCE = """\
You are analyzing a chain-of-thought (CoT) reasoning trace from an LLM solving a science (GPQA) question.

Identify "knot" segments: places where the reasoning gets stuck in circular thinking.
Specifically look for:
- Repeated re-examination of the same mechanism or hypothesis without progress
- Oscillation between answer choices with no new evidence
- Extended re-reading of the same constraint or premise in a loop
- Metacognitive uncertainty loops ("I think A, but wait maybe B, but actually A...") with no resolution

Return ONLY valid JSON:
{{"knots": [{{"start": <int>, "end": <int>}}, ...]}}

Rules:
- start/end are 0-indexed character positions in the text
- Each span must be >= 50 chars; spans must not overlap
- Return {{"knots": []}} if no knots found

Analyze only the first {window} characters:
{text_slice}
"""

_PROMPT_CODING = """\
You are analyzing a chain-of-thought (CoT) reasoning trace from an LLM solving a coding problem.

Identify "knot" segments: places where the reasoning goes in circles without making progress.
Specifically look for:
- Repeatedly debating the same algorithm choice (e.g. BFS vs DFS) without deciding
- Re-reading the same constraint or example in a loop
- Oscillating between two implementation approaches without committing
- Metacognitive evaluation loops with no new information

Return ONLY valid JSON:
{{"knots": [{{"start": <int>, "end": <int>}}, ...]}}

Rules:
- start/end are 0-indexed character positions in the text
- Each span must be >= 50 chars; spans must not overlap
- Return {{"knots": []}} if no knots found

Analyze only the first {window} characters:
{text_slice}
"""

DOMAIN_PROMPTS = {
    "math": _PROMPT_MATH,
    "science": _PROMPT_SCIENCE,
    "coding": _PROMPT_CODING,
}

# ── Token data arrays ──────────────────────────────────────────────────────────

TOKEN_ARRAYS = ["tok_conf", "tok_gini", "tok_logprob", "tok_neg_entropy", "tok_selfcert"]


# ── 1. Load data ──────────────────────────────────────────────────────────────

def load_data(cache_path: Path, n_sample: int = N_SAMPLE, seed: int = SEED) -> list[dict]:
    """Sample balanced correct/incorrect runs; load all 5 per-token signal arrays."""
    td = cache_path / "token_data"
    row_ptr = np.fromfile(td / "token_row_ptr.int64", dtype=np.int64)
    tok_ids = np.fromfile(td / "token_ids.int32",     dtype=np.int32)

    arrays = {
        name: np.fromfile(td / f"{name}.float32", dtype=np.float32)
        for name in TOKEN_ARRAYS
    }

    with open(cache_path / "evaluation_report.json") as f:
        report = json.load(f)

    all_runs: list[tuple[int, str, bool]] = []
    for prob in report["results"]:
        pid = str(prob["problem_id"])
        for run in prob.get("runs", []):
            sid = run.get("sample_id", run.get("run_index", -1))
            all_runs.append((int(sid), pid, bool(run.get("is_correct", False))))

    rng = np.random.default_rng(seed)
    correct   = [r for r in all_runs if r[2]]
    incorrect = [r for r in all_runs if not r[2]]
    half = n_sample // 2
    sampled = (
        [correct[i]   for i in rng.choice(len(correct),   min(half, len(correct)),   replace=False)] +
        [incorrect[i] for i in rng.choice(len(incorrect), min(half, len(incorrect)), replace=False)]
    )

    records = []
    for run_idx, pid, is_correct in sampled:
        s, e = int(row_ptr[run_idx]), int(row_ptr[run_idx + 1])
        rec = {
            "run_idx":    run_idx,
            "problem_id": pid,
            "is_correct": is_correct,
            "tokens":     tok_ids[s:e].copy(),
        }
        for name, arr in arrays.items():
            rec[name] = arr[s:e].copy()
        records.append(rec)
    return records


# ── 2. Decode + build offset map ───────────────────────────────────────────────

def decode_and_map(runs: list[dict], tokenizer) -> list[dict]:
    """Decode token IDs and build char-offset-per-token via fast tokenizer."""
    for r in tqdm(runs, desc="  Decode + offset map"):
        text = tokenizer.decode(r["tokens"].tolist(), skip_special_tokens=True)
        r["text"] = text
        # Re-tokenize to get per-token char offsets in the decoded string
        enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        r["offset_mapping"] = enc["offset_mapping"]  # list of (char_s, char_e)
        r["retoken_ids"]    = enc["input_ids"]
    return runs


# ── 3. Build token mask from char spans ────────────────────────────────────────

def char_spans_to_keep_mask(offset_mapping: list, knot_spans: list[dict]) -> np.ndarray:
    """Return bool array: True = keep token, False = knot token.
    Uses re-tokenized offset_mapping for char-level precision.
    """
    n = len(offset_mapping)
    keep = np.ones(n, dtype=bool)
    for span in knot_spans:
        s, e = span["start"], span["end"]
        for i, (ts, te) in enumerate(offset_mapping):
            if ts < e and te > s:  # token overlaps with knot span
                keep[i] = False
    return keep


def align_mask_to_orig(keep_mask_retoken: np.ndarray, n_orig: int) -> np.ndarray:
    """Align a mask for re-tokenized tokens to the length of original tokens.
    If lengths match, returns directly. Otherwise, interpolates proportionally.
    """
    n_re = len(keep_mask_retoken)
    if n_re == n_orig:
        return keep_mask_retoken
    if n_re == 0 or n_orig == 0:
        return np.ones(n_orig, dtype=bool)
    # Proportional mapping: token i in orig maps to token round(i * n_re / n_orig) in re
    scale = n_re / n_orig
    orig_indices = np.arange(n_orig)
    re_indices   = np.clip(np.round(orig_indices * scale).astype(int), 0, n_re - 1)
    return keep_mask_retoken[re_indices]


# ── 4. GLM knot detection ──────────────────────────────────────────────────────

def _call_glm_single(client, run: dict, prompt_template: str, retries: int = GLM_RETRIES) -> list[dict]:
    text_slice = run["text"][:TEXT_WINDOW]
    prompt = prompt_template.format(window=TEXT_WINDOW, text_slice=text_slice)
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=GLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=512,
            )
            raw = resp.choices[0].message.content.strip()
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if not m:
                return []
            data  = json.loads(m.group())
            knots = data.get("knots", [])
            text_len = len(run["text"])
            valid, last_end = [], -1
            for k in sorted(knots, key=lambda x: int(x.get("start", 0))):
                s, e = int(k.get("start", 0)), int(k.get("end", 0))
                if 0 <= s < e <= text_len and (e - s) >= 50 and s >= last_end:
                    valid.append({"start": s, "end": e})
                    last_end = e
            return valid
        except Exception:
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    return []


def detect_knots_glm(runs: list[dict], domain: str, max_workers: int = GLM_WORKERS) -> list[dict]:
    client = OpenAI(api_key=GLM_KEY, base_url=GLM_BASE)
    prompt_template = DOMAIN_PROMPTS[domain]
    results: dict[int, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_call_glm_single, client, r, prompt_template): r["run_idx"] for r in runs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"  GLM [{domain}]"):
            results[futures[fut]] = fut.result()
    for r in runs:
        r["knot_spans"] = results.get(r["run_idx"], [])
    return runs


# ── 5. Splice text ────────────────────────────────────────────────────────────

def splice_deknot(text: str, knot_spans: list[dict]) -> str:
    for span in sorted(knot_spans, key=lambda x: x["start"], reverse=True):
        s, e = span["start"], span["end"]
        text = text[:s] + " " + text[e:]
    return text


# ── 6. Feature computation ────────────────────────────────────────────────────

def _jaccard(a: set, b: set) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def traj_features_from_text(text: str) -> dict:
    words = text.split()
    if len(words) < SLICE_WORDS * 2:
        return {"traj_continuity": 0.5, "traj_novelty": 0.5, "traj_reflection_count": 0.0}
    slices = [set(words[i:i + SLICE_WORDS]) for i in range(0, len(words) - SLICE_WORDS + 1, SLICE_WORDS)]
    n = len(slices)
    cont = float(np.mean([_jaccard(slices[i], slices[i+1]) for i in range(n-1)])) if n > 1 else 0.5
    nov  = float(np.mean([1.0 - max(_jaccard(slices[i], slices[j]) for j in range(i)) for i in range(1, n)])) if n > 1 else 0.5
    refl = -float(sum(1 for i in range(n) for j in range(i+2, n) if _jaccard(slices[i], slices[j]) > REFL_THRESH))
    return {"traj_continuity": cont, "traj_novelty": nov, "traj_reflection_count": refl}


def compute_features(run: dict, keep_mask_orig: np.ndarray | None = None) -> dict:
    """Compute all 8 features (5 token-level + 3 trajectory).

    If keep_mask_orig is None, use full token arrays (original condition).
    Otherwise apply the mask to exclude knot-span tokens.
    """
    # Token-level features
    tok_feats = {}
    for name in TOKEN_ARRAYS:
        arr = run[name]
        if keep_mask_orig is not None and keep_mask_orig.sum() > 0:
            arr = arr[keep_mask_orig]
        tok_feats[f"{name}_mean"] = float(np.mean(arr)) if len(arr) > 0 else 0.0

    # Trajectory features from (optionally spliced) text
    if keep_mask_orig is not None:
        text = splice_deknot(run["text"], run["knot_spans"])
    else:
        text = run["text"]
    traj = traj_features_from_text(text)

    return {**tok_feats, **traj}


FEATURE_KEYS = [f"{n}_mean" for n in TOKEN_ARRAYS] + [
    "traj_continuity", "traj_novelty", "traj_reflection_count"
]


# ── 7. Cross-validated AUROC ──────────────────────────────────────────────────

def compute_auroc_cv(feature_dicts: list[dict], labels: list[int], groups: list[str]) -> float:
    X = np.array([[fd[k] for k in FEATURE_KEYS] for fd in feature_dicts])
    y = np.array(labels)
    g = np.array(groups)
    gkf = GroupKFold(n_splits=min(5, len(set(groups))))
    aurocs = []
    for tr, te in gkf.split(X, y, g):
        if len(np.unique(y[te])) < 2:
            continue
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        clf = LogisticRegression(class_weight="balanced", C=0.5, max_iter=1000, random_state=42)
        clf.fit(Xtr, y[tr])
        aurocs.append(roc_auc_score(y[te], clf.predict_proba(Xte)[:, 1]))
    return float(np.mean(aurocs)) if aurocs else float("nan")


# ── 8. Main ────────────────────────────────────────────────────────────────────

def run_domain(domain: str, tokenizer) -> dict:
    cfg = DOMAINS[domain]
    print(f"\n{'='*60}")
    print(f"Domain: {domain.upper()}")
    print(f"Cache:  {cfg['cache_path'].name}")
    print(f"{'='*60}")

    print("[1/5] Loading data...")
    runs = load_data(cfg["cache_path"])
    nc = sum(r["is_correct"] for r in runs)
    print(f"  {len(runs)} runs: {nc} correct, {len(runs)-nc} incorrect")

    print("[2/5] Decoding + building offset maps...")
    runs = decode_and_map(runs, tokenizer)

    print("[3/5] Detecting knots (GLM)...")
    runs = detect_knots_glm(runs, domain)
    n_knotted = sum(1 for r in runs if r["knot_spans"])
    total_chars = sum(sum(s["end"]-s["start"] for s in r["knot_spans"]) for r in runs)
    knot_in_correct   = sum(1 for r in runs if r["is_correct"] and r["knot_spans"])
    knot_in_incorrect = sum(1 for r in runs if not r["is_correct"] and r["knot_spans"])
    n_c = sum(r["is_correct"] for r in runs)
    n_i = len(runs) - n_c
    print(f"  Knotted: {n_knotted}/{len(runs)}  |  "
          f"correct={knot_in_correct}/{n_c} ({knot_in_correct/max(1,n_c):.0%})  "
          f"incorrect={knot_in_incorrect}/{n_i} ({knot_in_incorrect/max(1,n_i):.0%})")
    print(f"  Total chars to remove: {total_chars:,}")

    print("[4/5] Computing features...")
    orig_features, deknot_features = [], []
    for r in tqdm(runs, desc="  Features"):
        # Build token-level keep mask
        if r["knot_spans"]:
            mask_re = char_spans_to_keep_mask(r["offset_mapping"], r["knot_spans"])
            mask    = align_mask_to_orig(mask_re, len(r["tokens"]))
        else:
            mask = None

        orig_features.append(compute_features(r, keep_mask_orig=None))
        deknot_features.append(compute_features(r, keep_mask_orig=mask))

    print("[5/5] Cross-validated AUROC...")
    labels = [int(r["is_correct"]) for r in runs]
    groups = [r["problem_id"] for r in runs]
    auroc_orig   = compute_auroc_cv(orig_features,   labels, groups)
    auroc_deknot = compute_auroc_cv(deknot_features, labels, groups)
    delta = auroc_deknot - auroc_orig

    tag = "POSITIVE (helps)" if delta > 0.02 else ("HURTS" if delta < -0.02 else "NEUTRAL (no change)")
    print(f"\n  Original   AUROC: {auroc_orig:.4f}")
    print(f"  De-knotted AUROC: {auroc_deknot:.4f}")
    print(f"  Delta:           {delta:+.4f}  [{tag}]")

    return {
        "domain":             domain,
        "n_runs":             len(runs),
        "n_knotted":          n_knotted,
        "knot_rate_correct":  round(knot_in_correct / max(1, n_c), 3),
        "knot_rate_incorrect":round(knot_in_incorrect / max(1, n_i), 3),
        "total_chars_removed":total_chars,
        "auroc_original":     round(auroc_orig, 4),
        "auroc_deknot":       round(auroc_deknot, 4),
        "delta":              round(delta, 4),
        "verdict":            tag,
    }


def main() -> None:
    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    print(f"  {type(tok).__name__}, is_fast={tok.is_fast}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for domain in ["math", "science", "coding"]:
        row = run_domain(domain, tok)
        results.append(row)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Domain':<10} {'Orig':>6} {'Deknot':>7} {'Delta':>7}  {'Verdict'}")
    for r in results:
        print(f"{r['domain']:<10} {r['auroc_original']:>6.4f} {r['auroc_deknot']:>7.4f} "
              f"{r['delta']:>+7.4f}  {r['verdict']}")

    # Save
    summary_path = OUT_DIR / "deknot_alldomains_v2.csv"
    fieldnames = list(results[0].keys())
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
