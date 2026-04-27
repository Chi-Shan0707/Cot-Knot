# CoT + Knot Review Packets v1

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
