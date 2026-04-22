"""
Verify extracted math knot spans with a stricter GLM veto
=========================================================
Given extracted span quotes, ask GLM whether the quote itself is strong enough
to count as an explicit knot span. This is a high-precision second-pass filter.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_glm_math_knot_labeling import (  # noqa: E402
    BASE_RETRY_DELAY,
    GLM_ENDPOINT,
    GLM_MODEL,
    MAX_RETRIES,
    RateLimiter,
    get_api_key,
    recommend_workers,
)


SYSTEM_PROMPT = (
    "You are a strict verifier for quoted math reasoning spans. "
    "Only approve a span if the quote itself contains explicit local state break. "
    "Respond in valid JSON only."
)

USER_TEMPLATE = """\
Decide whether the quoted span below is, by itself, strong enough evidence of an explicit math reasoning knot.

[QUOTE]
{quote}

[PROPOSED SYMPTOM]
{symptom}

[PROPOSED EXPLANATION]
{why}

Approve ONLY if the quote itself shows one of:
1) explicit contradiction,
2) incompatible redefinition of the same symbol/object,
3) switching cases without closing the prior case,
4) repeated repair that still leaves the active state unclear.

Reject if the quote is only:
- coordinate setup
- ordinary self-check
- method exploration
- equation restatement
- an incomplete sentence
- a single "wait" without lasting inconsistency

NEGATIVE EXAMPLES:
- "Let A=(0,0), B=(107,0), C=(107,16), D=(0,16). Wait, DC has length 107, so this setup is consistent."
- "Let N = 1000A + 100B + 10C + D."

POSITIVE EXAMPLES:
- "Assume n is even. Then n=2k+1."
- "Case 1: x>0 ... therefore x<0 in this case."
- "So a=b. Wait, now let a be the index and b the value."

Return EXACTLY this JSON:
{{
  "valid_explicit_knot": "<yes|no>",
  "evidence_type": "<explicit_contradiction|double_definition|case_leak|unresolved_repair|none>",
  "confidence": "<high|medium|low>",
  "reason": "<one-sentence reason>"
}}
"""


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
        "max_tokens": 400,
    }
    for attempt in range(1, MAX_RETRIES + 1):
        delay = BASE_RETRY_DELAY * (2 ** (attempt - 1))
        try:
            response = requests.post(GLM_ENDPOINT, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()
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


def extract_content(response: dict) -> str:
    try:
        return response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""


def parse_content(content: str) -> tuple[dict, str | None]:
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


def iter_spans(input_jsonl: Path):
    with open(input_jsonl) as handle:
        for line in handle:
            record = json.loads(line)
            parsed = record.get("glm_parsed", {}) or {}
            spans = parsed.get("explicit_knot_spans", [])
            if not isinstance(spans, list):
                continue
            for idx, span in enumerate(spans):
                if not isinstance(span, dict):
                    continue
                yield {
                    "dataset": record["dataset"],
                    "problem_id": record["problem_id"],
                    "run_index": int(record["run_index"]),
                    "is_correct": int(record["is_correct"]),
                    "span_idx": idx,
                    "quote": str(span.get("quote", "")),
                    "symptom": str(span.get("symptom", "")),
                    "why": str(span.get("why_it_is_a_knot", "")),
                }


def build_prompt(span: dict) -> str:
    return USER_TEMPLATE.format(
        quote=span["quote"],
        symptom=span["symptom"],
        why=span["why"],
    )


def main():
    parser = argparse.ArgumentParser(description="Verify math knot spans")
    parser.add_argument("--input-jsonl", type=str, required=True)
    parser.add_argument("--out-jsonl", type=str, required=True)
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--model", type=str, default=GLM_MODEL)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--requests-per-second", type=float, default=2.0)
    parser.add_argument("--max-spans", type=int, default=0)
    args = parser.parse_args()

    api_key = get_api_key(args.api_key)
    input_jsonl = Path(args.input_jsonl)
    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    spans = list(iter_spans(input_jsonl))
    if args.max_spans > 0:
        spans = spans[: args.max_spans]

    workers, cpu_info = recommend_workers()
    if args.workers > 0:
        workers = args.workers
    print(json.dumps(cpu_info, ensure_ascii=False))
    print(f"Verifying {len(spans)} spans")

    rate_limiter = RateLimiter(args.requests_per_second)
    counts = defaultdict(int)

    def run_one(span: dict) -> dict:
        rate_limiter.wait()
        response = call_glm(build_prompt(span), api_key, args.model)
        content = extract_content(response)
        parsed, parse_error = parse_content(content)
        row = {
            **span,
            "glm_raw_content": content,
            "glm_parsed": parsed,
            "parse_error": parse_error,
        }
        return row

    rows = []
    with futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for idx, row in enumerate(executor.map(run_one, spans), 1):
            rows.append(row)
            verdict = (row.get("glm_parsed") or {}).get("valid_explicit_knot", "")
            if verdict == "yes":
                counts["yes"] += 1
            elif verdict == "no":
                counts["no"] += 1
            else:
                counts["other"] += 1
            if idx % 25 == 0:
                print(f"[{idx:5d}/{len(spans)}] yes={counts['yes']} no={counts['no']} other={counts['other']}")

    with open(out_jsonl, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved -> {out_jsonl}")
    print(f"Counts: {dict(counts)}")


if __name__ == "__main__":
    main()
