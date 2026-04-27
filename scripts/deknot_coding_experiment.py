#!/usr/bin/env python3
"""
De-knotting experiment for coding CoT traces.

Research question: Does physically removing "knot" (circular/looping reasoning) segments
from coding CoT traces improve verifier AUROC for predicting run correctness?

Both outcomes are informative:
  POSITIVE (AUROC improves) → knots are verifier noise; removing them cleans the signal
  NEGATIVE (AUROC flat)     → coding failure is fundamental; execution state not observable

Usage:
    cd /home/jovyan/work/NAD_Next && source .venv/bin/activate
    python /home/jovyan/work/SVDomain/workshop/cotknot/scripts/deknot_coding_experiment.py
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

# ── Paths & constants ────────────────────────────────────────────────────────

CACHE_ROOT = Path(
    "/home/jovyan/public-ro/MUI_HUB/cache"
    "/DeepSeek-R1-0528-Qwen3-8B/livecodebench_v5"
    "/cache_neuron_output_1_act_no_rms_20251127_032808"
)
TOKENIZER_PATH = "/home/jovyan/public-ro/model/DeepSeek-R1-0528-Qwen3-8B"
GLM_KEY  = "49bbfdfe02384b74b6882da123bd6ee6.Xp4qZQTqnS3s5130"
GLM_BASE = "https://open.bigmodel.cn/api/paas/v4/"
GLM_MODEL = "glm-4-flash"
OUT_DIR  = Path("/home/jovyan/work/SVDomain/workshop/cotknot/results/tables")

N_SAMPLE     = 200   # 100 correct + 100 incorrect
SEED         = 42
SLICE_WORDS  = 40    # words per trajectory slice
REFL_THRESH  = 0.30  # Jaccard threshold for reflection detection
GLM_WORKERS  = 8
GLM_RETRIES  = 2

# ── 1. Load data ─────────────────────────────────────────────────────────────

def load_data(n_sample: int = N_SAMPLE, seed: int = SEED) -> list[dict]:
    """Sample n_sample/2 correct + n_sample/2 incorrect coding runs."""
    td = CACHE_ROOT / "token_data"
    row_ptr = np.fromfile(td / "token_row_ptr.int64", dtype=np.int64)
    tok_ids = np.fromfile(td / "token_ids.int32",     dtype=np.int32)
    tok_conf = np.fromfile(td / "tok_conf.float32",   dtype=np.float32)

    with open(CACHE_ROOT / "evaluation_report.json") as f:
        report = json.load(f)

    # Build flat list: (run_idx, problem_id, is_correct)
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
    sampled_c = [correct[i]   for i in rng.choice(len(correct),   min(half, len(correct)),   replace=False)]
    sampled_i = [incorrect[i] for i in rng.choice(len(incorrect), min(half, len(incorrect)), replace=False)]
    sampled   = sampled_c + sampled_i

    records = []
    for run_idx, pid, is_correct in sampled:
        start, end = int(row_ptr[run_idx]), int(row_ptr[run_idx + 1])
        records.append({
            "run_idx":    run_idx,
            "problem_id": pid,
            "is_correct": is_correct,
            "tokens":     tok_ids[start:end].copy(),
            "conf":       tok_conf[start:end].copy(),
        })
    return records


# ── 2. Decode tokens → text ──────────────────────────────────────────────────

def decode_runs(runs: list[dict], tokenizer_path: str = TOKENIZER_PATH) -> list[dict]:
    print("  Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    for r in tqdm(runs, desc="  Decoding"):
        r["text"] = tok.decode(r["tokens"].tolist(), skip_special_tokens=True)
    return runs


# ── 3. GLM knot-span detection ───────────────────────────────────────────────

_KNOT_PROMPT = """\
You are analyzing a chain-of-thought (CoT) reasoning trace from an LLM solving a coding problem.

Identify "knot" segments: places where the reasoning goes in circles, repeats the same idea, \
or gets stuck deliberating without making progress \
(e.g. repeatedly debating BFS vs DFS without deciding, or re-reading the same constraint loop).

Return ONLY valid JSON in this exact format:
{{"knots": [{{"start": <int>, "end": <int>}}, ...]}}

Rules:
- start/end are 0-indexed character positions in the text below
- Each span must be >= 50 chars
- Spans must not overlap
- Return {{"knots": []}} if no knots found

Analyze only the first 3000 characters of this trace:
{text_slice}
"""

def _call_glm_single(client: OpenAI, run: dict, retries: int = GLM_RETRIES) -> list[dict]:
    text_slice = run["text"][:3000]
    prompt = _KNOT_PROMPT.format(text_slice=text_slice)
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=GLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=512,
            )
            raw = resp.choices[0].message.content.strip()
            # Extract JSON even if there's surrounding text
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if not m:
                return []
            data = json.loads(m.group())
            knots = data.get("knots", [])
            # Validate spans
            valid = []
            text_len = len(run["text"])
            for k in knots:
                s, e = int(k.get("start", 0)), int(k.get("end", 0))
                if 0 <= s < e <= text_len and (e - s) >= 50:
                    valid.append({"start": s, "end": e})
            # Remove overlaps (keep first)
            valid.sort(key=lambda x: x["start"])
            deduped = []
            last_end = -1
            for k in valid:
                if k["start"] >= last_end:
                    deduped.append(k)
                    last_end = k["end"]
            return deduped
        except Exception as exc:
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
            else:
                return []
    return []


def detect_knots_glm(runs: list[dict], max_workers: int = GLM_WORKERS) -> list[dict]:
    client = OpenAI(api_key=GLM_KEY, base_url=GLM_BASE)
    results: dict[int, list[dict]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_call_glm_single, client, r): r["run_idx"] for r in runs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="  GLM knot detection"):
            run_idx = futures[fut]
            results[run_idx] = fut.result()

    for r in runs:
        r["knot_spans"] = results.get(r["run_idx"], [])
    return runs


# ── 4. Splice / de-knot ──────────────────────────────────────────────────────

def splice_deknot(text: str, knot_spans: list[dict]) -> str:
    """Remove knot character spans from text (process last→first to preserve offsets)."""
    spans = sorted(knot_spans, key=lambda x: x["start"], reverse=True)
    for span in spans:
        s, e = span["start"], span["end"]
        text = text[:s] + " " + text[e:]
    return text


# ── 5. Trajectory feature computation ───────────────────────────────────────

def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def compute_traj_features(text: str, tok_conf_mean: float) -> dict:
    """Compute 4 trajectory-based features from text."""
    words = text.split()
    if len(words) < SLICE_WORDS * 2:
        # Too short for meaningful trajectory analysis
        return {
            "traj_continuity":       0.5,
            "traj_novelty":          0.5,
            "traj_reflection_count": 0.0,
            "tok_conf_prefix":       tok_conf_mean,
        }

    # Split into non-overlapping slices of SLICE_WORDS words
    slices = [
        set(words[i : i + SLICE_WORDS])
        for i in range(0, len(words) - SLICE_WORDS + 1, SLICE_WORDS)
    ]
    n = len(slices)

    # traj_continuity: mean Jaccard between consecutive slices
    continuity_vals = [_jaccard(slices[i], slices[i + 1]) for i in range(n - 1)]
    traj_continuity = float(np.mean(continuity_vals)) if continuity_vals else 0.5

    # traj_novelty: for each slice i>0, (1 - max Jaccard vs all prior slices)
    novelty_vals = []
    for i in range(1, n):
        max_sim = max(_jaccard(slices[i], slices[j]) for j in range(i))
        novelty_vals.append(1.0 - max_sim)
    traj_novelty = float(np.mean(novelty_vals)) if novelty_vals else 0.5

    # traj_reflection_count: count non-adjacent pairs (gap>=2) with Jaccard > threshold
    # Negated so that more reflection = lower (more negative) value (consistent with paper)
    reflection_count = 0
    for i in range(n):
        for j in range(i + 2, n):
            if _jaccard(slices[i], slices[j]) > REFL_THRESH:
                reflection_count += 1
    traj_reflection_count = -float(reflection_count)

    return {
        "traj_continuity":       traj_continuity,
        "traj_novelty":          traj_novelty,
        "traj_reflection_count": traj_reflection_count,
        "tok_conf_prefix":       tok_conf_mean,
    }


# ── 6. Cross-validated AUROC ─────────────────────────────────────────────────

FEATURE_KEYS = ["traj_continuity", "traj_novelty", "traj_reflection_count", "tok_conf_prefix"]


def compute_auroc_cv(
    feature_dicts: list[dict],
    labels: list[int],
    groups: list[str],
    n_splits: int = 5,
) -> float:
    X = np.array([[fd[k] for k in FEATURE_KEYS] for fd in feature_dicts])
    y = np.array(labels)
    g = np.array(groups)

    gkf = GroupKFold(n_splits=min(n_splits, len(set(groups))))
    aurocs = []
    for train_idx, test_idx in gkf.split(X, y, g):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        if len(np.unique(y_te)) < 2:
            continue
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        clf = LogisticRegression(class_weight="balanced", C=0.5, max_iter=1000, random_state=42)
        clf.fit(X_tr_s, y_tr)
        proba = clf.predict_proba(X_te_s)[:, 1]
        aurocs.append(roc_auc_score(y_te, proba))

    return float(np.mean(aurocs)) if aurocs else float("nan")


# ── 7. Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("De-knotting Experiment — Coding CoT Traces")
    print("=" * 60)

    # Step 1: Load
    print("\n[1/6] Loading data...")
    runs = load_data()
    n_correct = sum(r["is_correct"] for r in runs)
    print(f"  Sampled {len(runs)} runs: {n_correct} correct, {len(runs)-n_correct} incorrect")

    # Step 2: Decode
    print("\n[2/6] Decoding token IDs → text...")
    runs = decode_runs(runs)
    avg_len = np.mean([len(r["text"]) for r in runs])
    print(f"  Average trace length: {avg_len:.0f} chars")

    # Step 3: GLM knot detection
    print(f"\n[3/6] Detecting knots with GLM ({GLM_MODEL}, {GLM_WORKERS} workers)...")
    runs = detect_knots_glm(runs)
    n_knotted = sum(1 for r in runs if r["knot_spans"])
    total_chars_removed = sum(
        sum(s["end"] - s["start"] for s in r["knot_spans"]) for r in runs
    )
    print(f"  Knots found in {n_knotted}/{len(runs)} runs ({n_knotted/len(runs):.1%})")
    print(f"  Total chars to be removed: {total_chars_removed:,}")
    if n_knotted > 0:
        avg_knot_len = total_chars_removed / n_knotted
        print(f"  Avg chars removed per knotted run: {avg_knot_len:.0f}")

    # Step 4 & 5: Compute features
    print("\n[4/6] Computing features on original traces...")
    orig_features = [
        compute_traj_features(r["text"], float(np.mean(r["conf"]))) for r in runs
    ]

    print("\n[5/6] Splicing + computing features on de-knotted traces...")
    deknot_features = []
    for r in tqdm(runs, desc="  De-knotting"):
        clean = splice_deknot(r["text"], r["knot_spans"])
        deknot_features.append(compute_traj_features(clean, float(np.mean(r["conf"]))))

    # Step 6: AUROC comparison
    print("\n[6/6] Computing cross-validated AUROCs...")
    labels = [int(r["is_correct"]) for r in runs]
    groups = [r["problem_id"] for r in runs]

    auroc_orig   = compute_auroc_cv(orig_features,   labels, groups)
    auroc_deknot = compute_auroc_cv(deknot_features, labels, groups)
    delta = auroc_deknot - auroc_orig

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Original   AUROC: {auroc_orig:.4f}")
    print(f"  De-knotted AUROC: {auroc_deknot:.4f}")
    print(f"  Delta:            {delta:+.4f}")
    print()
    if delta > 0.02:
        print(">> POSITIVE: Knots are verifier noise — removing them improves discrimination.")
        print("   Implication: knot segments contaminate text-based features in coding traces.")
    elif delta < -0.02:
        print(">> UNEXPECTED: De-knotting *hurts* discrimination.")
        print("   Implication: knot segments may actually carry signal (even if noisy).")
    else:
        print(">> NEGATIVE: De-knotting does NOT help (|Δ| < 0.02).")
        print("   Implication: coding failure is fundamental — execution state not observable")
        print("   from text surface regardless of knot removal.")

    # Save summary CSV
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "deknot_coding_auroc_comparison.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["condition", "auroc", "n_runs", "n_knotted", "total_chars_removed"]
        )
        w.writeheader()
        w.writerow({
            "condition": "original", "auroc": round(auroc_orig, 4),
            "n_runs": len(runs), "n_knotted": n_knotted,
            "total_chars_removed": 0,
        })
        w.writerow({
            "condition": "de_knotted", "auroc": round(auroc_deknot, 4),
            "n_runs": len(runs), "n_knotted": n_knotted,
            "total_chars_removed": total_chars_removed,
        })

    # Save per-run stats
    stats_path = OUT_DIR / "deknot_run_stats.csv"
    per_run_rows = []
    for r, of, df in zip(runs, orig_features, deknot_features):
        per_run_rows.append({
            "run_idx":               r["run_idx"],
            "problem_id":            r["problem_id"],
            "is_correct":            int(r["is_correct"]),
            "n_knots":               len(r["knot_spans"]),
            "chars_removed":         sum(s["end"] - s["start"] for s in r["knot_spans"]),
            "orig_len":              len(r["text"]),
            "orig_traj_reflection":  of["traj_reflection_count"],
            "deknot_traj_reflection": df["traj_reflection_count"],
            "orig_traj_continuity":  of["traj_continuity"],
            "deknot_traj_continuity": df["traj_continuity"],
        })

    with open(stats_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_run_rows[0].keys()))
        w.writeheader()
        w.writerows(per_run_rows)

    print(f"\nSaved:")
    print(f"  {summary_path}")
    print(f"  {stats_path}")


if __name__ == "__main__":
    main()
