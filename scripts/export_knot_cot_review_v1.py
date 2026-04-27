"""
Export human-readable CoT + knot review packets
==============================================

Create a review folder that pairs each v4 knot-labeled run with:
- problem snippet
- natural-language CoT excerpt (the same style used during labeling)
- final knot label and raw GLM label

Outputs are intended for manual reading, not model training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from knot_glm_common import build_trace_excerpt
from knot_v4_configs import DOMAIN_CONFIGS


REVIEW_DIR = Path(__file__).resolve().parent.parent / "results" / "cot_knot_review_v1"


def load_source_lookup(report_paths: dict[str, Path]) -> dict[tuple[str, str, int], dict]:
    lookup: dict[tuple[str, str, int], dict] = {}
    for dataset, report_path in report_paths.items():
        report = json.loads(report_path.read_text())
        for problem in report["results"]:
            problem_id = str(problem["problem_id"])
            prompt = str(problem.get("prompt") or "")
            for run in problem.get("runs", []):
                run_index = int(run["run_index"])
                lookup[(dataset, problem_id, run_index)] = {
                    "dataset": dataset,
                    "problem_id": problem_id,
                    "run_index": run_index,
                    "prompt": prompt,
                    "actual_prompt": str(run.get("actual_prompt") or ""),
                    "generated_text": str(run.get("generated_text") or ""),
                }
    return lookup


def load_domain_review_frame(domain: str, excerpt_chars: int) -> pd.DataFrame:
    cfg = DOMAIN_CONFIGS[domain]
    labels_path = REVIEW_DIR.parent / "tables" / f"glm_{domain}_knot_labels_v4.csv"
    labels = pd.read_csv(labels_path)
    labels["dataset"] = labels["dataset"].astype(str)
    labels["problem_id"] = labels["problem_id"].astype(str)
    labels["run_index"] = labels["run_index"].astype(int)
    labels["problem_key"] = labels["problem_key"].astype(str)

    source_lookup = load_source_lookup(cfg.report_paths)

    rows: list[dict] = []
    for _, row in labels.iterrows():
        key = (str(row["dataset"]), str(row["problem_id"]), int(row["run_index"]))
        source = source_lookup.get(key)
        if source is None:
            continue
        actual_prompt = source["actual_prompt"] or source["prompt"]
        think_excerpt, think_total_chars = build_trace_excerpt(source["generated_text"], max_chars=excerpt_chars)

        review_bucket = "clean_negative"
        if int(row["parse_ok"]) == 0:
            review_bucket = "parse_error"
        elif str(row["raw_knot_present"]) == "yes" and int(row["knot_present_bin"]) == 0:
            review_bucket = "filtered_raw_positive"
        elif int(row["knot_present_bin"]) == 1:
            review_bucket = "accepted_knot"

        rows.append(
            {
                "domain": domain,
                "dataset": source["dataset"],
                "problem_id": source["problem_id"],
                "problem_key": row["problem_key"],
                "run_index": source["run_index"],
                "is_correct": int(row["is_correct"]),
                "parse_ok": int(row["parse_ok"]),
                "review_bucket": review_bucket,
                "problem_snippet": actual_prompt[:1000],
                "think_excerpt": think_excerpt,
                "think_total_chars": think_total_chars,
                "raw_knot_present": row["raw_knot_present"],
                "raw_knot_severity": row["raw_knot_severity"],
                "raw_knot_symptoms": row["raw_knot_symptoms"],
                "protocol_filter": row["protocol_filter"],
                "knot_present": row["knot_present"],
                "knot_present_bin": int(row["knot_present_bin"]),
                "knot_severity": int(row["knot_severity"]),
                "primary_trigger": row["primary_trigger"],
                "trace_strategy": row["trace_strategy"],
                "reversal_count": row["reversal_count"],
                "state_consistency": row["state_consistency"],
                "annotator_confidence": row["annotator_confidence"],
                "knot_symptoms": row["knot_symptoms"],
                "knot_quote": row["knot_quote"],
                "open_diagnosis": row["open_diagnosis"],
            }
        )

    return pd.DataFrame(rows).sort_values(["review_bucket", "dataset", "problem_id", "run_index"]).reset_index(drop=True)


def write_readme(out_dir: Path):
    text = """# CoT + Knot Review Packets v1

这个文件夹是给人工阅读准备的。

## 你先看哪些文件

- `accepted_knot_runs_v4.csv`：最终被协议接受为 knot 的 run
- `filtered_raw_positive_runs_v4.csv`：GLM 原始说有 knot，但最终协议过滤掉的 run
- `all_labeled_runs_v4.csv`：全部已标注 run
- `spotlight_examples_v4.md`：按域挑出来的代表性例子

## 关键列

- `problem_snippet`：题目/提示片段
- `think_excerpt`：用于标注的自然语言 CoT 摘录
- `review_bucket`：
  - `accepted_knot`
  - `filtered_raw_positive`
  - `clean_negative`
  - `parse_error`
- `raw_knot_present`：GLM 原始判断
- `knot_present`：协议过滤后的最终判断
- `protocol_filter`：为什么被接受/过滤
- `knot_symptoms`：最终症状
- `knot_quote`：标注时抓到的局部 knot 片段
- `open_diagnosis`：自由文本诊断

## 阅读建议

如果你想先看“最像东西”的例子：

1. 先看 `accepted_knot_runs_v4.csv`
2. 再看 `filtered_raw_positive_runs_v4.csv`
3. 比较同一域里两者的差别

这样最容易判断：哪些 run 真的是 knot，哪些只是“看起来乱但证据不够硬”。
"""
    (out_dir / "README.md").write_text(text)


def write_spotlight(frames: dict[str, pd.DataFrame], out_dir: Path, n_per_bucket: int = 5):
    lines = [
        "# Spotlight CoT Examples v1",
        "",
        "下面每个 domain 都给两组例子：",
        "- `accepted_knot`：最终被接受的 knot",
        "- `filtered_raw_positive`：原始 GLM 判正，但被协议过滤掉",
        "",
    ]
    for domain, df in frames.items():
        lines.append(f"## {domain}")
        lines.append("")
        for bucket in ["accepted_knot", "filtered_raw_positive"]:
            sub = df[df["review_bucket"] == bucket].head(n_per_bucket)
            lines.append(f"### {bucket}")
            lines.append("")
            if sub.empty:
                lines.append("- (none)")
                lines.append("")
                continue
            for _, row in sub.iterrows():
                lines.extend(
                    [
                        f"- **ID**: `{row['dataset']} / {row['problem_id']} / run {row['run_index']}` | `correct={row['is_correct']}` | `final_knot={row['knot_present']}` | `filter={row['protocol_filter']}`",
                        f"  - **Symptoms**: `{row['knot_symptoms']}`",
                        f"  - **Quote**: {str(row['knot_quote'])[:300]}",
                        f"  - **Diagnosis**: {str(row['open_diagnosis'])[:400]}",
                        f"  - **CoT Excerpt**: {str(row['think_excerpt'])[:1200].replace(chr(10), ' ')}",
                        "",
                    ]
                )
        lines.append("")
    (out_dir / "spotlight_examples_v4.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Export human-readable CoT + knot review packets")
    parser.add_argument("--domains", nargs="+", default=["math", "science", "coding"], choices=sorted(DOMAIN_CONFIGS))
    parser.add_argument("--excerpt-chars", type=int, default=12000)
    args = parser.parse_args()

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.DataFrame] = {}
    all_frames = []
    for domain in args.domains:
        df = load_domain_review_frame(domain, excerpt_chars=args.excerpt_chars)
        frames[domain] = df
        all_frames.append(df)
        df.to_csv(REVIEW_DIR / f"{domain}_labeled_runs_v4.csv", index=False)
        df[df["review_bucket"] == "accepted_knot"].to_csv(REVIEW_DIR / f"{domain}_accepted_knot_runs_v4.csv", index=False)
        df[df["review_bucket"] == "filtered_raw_positive"].to_csv(
            REVIEW_DIR / f"{domain}_filtered_raw_positive_runs_v4.csv", index=False
        )

    all_df = pd.concat(all_frames, ignore_index=True)
    all_df.to_csv(REVIEW_DIR / "all_labeled_runs_v4.csv", index=False)
    all_df[all_df["review_bucket"] == "accepted_knot"].to_csv(REVIEW_DIR / "accepted_knot_runs_v4.csv", index=False)
    all_df[all_df["review_bucket"] == "filtered_raw_positive"].to_csv(
        REVIEW_DIR / "filtered_raw_positive_runs_v4.csv", index=False
    )

    write_readme(REVIEW_DIR)
    write_spotlight(frames, REVIEW_DIR)

    print(f"Saved review folder -> {REVIEW_DIR}")
    print("Files:")
    for path in sorted(REVIEW_DIR.glob("*")):
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()
