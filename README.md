<div align="center">

# Cot-Knot

### Different Kinds of Knots in Chain-of-Thought

<p>
  <img src="https://img.shields.io/badge/status-workshop%20paper-blue" alt="status" />
  <img src="https://img.shields.io/badge/focus-CoT%20quality%20signals-6f42c1" alt="focus" />
  <img src="https://img.shields.io/badge/theme-math%20vs%20coding-0a7e5a" alt="theme" />
  <img src="https://img.shields.io/badge/license-Apache--2.0-orange" alt="license" />
</p>

*Self-correction is not a universal quality signal.*

[Paper Overview](#paper-overview) ·
[Main Claims](#main-claims) ·
[Knot Taxonomy](#knot-taxonomy) ·
[Repository Roadmap](#repository-roadmap) ·
[Citation](#citation)

</div>

`Cot-Knot` is a paper repository centered on one claim:

> superficially similar self-corrections in chain-of-thought do not necessarily mean the same thing across domains.

---

## Paper Overview

This project studies when an apparently self-corrective chain-of-thought trace actually carries reliable information about answer quality. The repository is organized as a paper-facing workspace for the **Cot-Knot** hypothesis: that chain-of-thought quality signals are meaningful only when the underlying local state dynamics are interpreted in a domain-appropriate way.

The core hypothesis is that **math and coding contain different kinds of knots**:

- **Math knots** are often explicit, text-visible local state breaks.
- **Coding knots** are often execution-semantic breaks, where the prose stops faithfully tracking latent program state.
- Therefore, the same surface pattern of “revision” or “self-correction” should not be assumed to have the same semantics across domains.

In other words, this repository asks a narrow but important question:

> when a model appears to “notice and repair” its own reasoning, is that event actually a stable signal of correctness?

The paper’s current answer is **no, not universally**. The meaning of self-correction depends on the domain and, in later sections of the project, also on the posttraining regime.

### Research question

The central research question is:

> **When does self-correction in chain-of-thought behave like a stable quality signal, and when does it stop meaning the same thing?**

The current repository treats this as a measurement and interpretation problem rather than a pure modeling problem. The aim is not only to score traces, but to understand what the apparent corrective behavior is actually doing.

## Main claims

The current paper direction centers on four linked ideas:

- **Domain-conditioned CoT quality signals**
- **Different knot types in math and coding**
- **Local state breaks as mechanism evidence**
- **RL warning signs under score drift and operating-point mismatch**

In the current framing, the paper makes three empirical claims:

1. **Compact CoT-based quality probes are strong in math, reasonable in science, and weak in coding.**
2. **Math and coding exhibit different knot types rather than one universal self-correction phenomenon.**
3. **RL posttraining can preserve ranking signal while breaking the original binary operating point of an SFT-era verifier.**

### High-level picture

| Domain | What the trace often reveals | What the knot tends to mean |
|---|---|---|
| **Math** | Explicit local state break | A visible repairable reasoning disruption |
| **Science** | Weaker but still useful process signal | A less concentrated quality signal |
| **Coding** | Surface revision with weak correctness coupling | Possible failure to faithfully track execution state |

## Knot taxonomy

In this repository, a **knot** is a mechanism-level local entanglement or break in reasoning state.

Typical examples include:

- explicit contradiction
- incompatible redefinition
- leaked case frame
- unresolved repair
- lost execution state
- patchy backtracking
- branch, loop, or index-state inconsistency

The important claim is not merely that knots occur, but that **their semantics depend on domain**.

### Working interpretation

- **Math**: a knot can be explicit, local, and still compatible with successful recovery.
- **Coding**: a similar-looking correction can mask a deeper execution-state failure.
- **Takeaway**: the same textual behavior can correspond to different underlying mechanisms.

### Methodological stance

This repository takes a deliberately conservative stance on interpretation:

- it separates **full-scale quantitative signal** from **pilot mechanism evidence**
- it treats annotation-derived knot labels as mechanism probes rather than universal ground truth
- it emphasizes **domain-conditioned interpretation** over universal verifier narratives
- it treats RL warning results primarily as evidence of **operating-point failure under regime shift**

## Repository roadmap

This repository is intended to hold the paper-facing materials for the project, including:

- workshop paper drafts
- figure assets for knot/domain comparisons
- annotation guidelines for local state breaks
- concise result summaries and paper tables
- minimal scripts or artifacts needed to support the workshop claims

### Intended structure

As the repository matures, it is expected to organize around:

- `paper/` for manuscript sources
- `figures/` for paper-ready visual assets
- `notes/` for concise paper-facing research notes
- `annotations/` for local state break guidelines or examples
- `artifacts/` for minimal supporting tables or metadata

## Status

This repository is in an early paper-focused stage.

The current version establishes the framing, identity, and licensing of the project. Paper files, figures, and supporting artifacts will be added incrementally.

## License

This repository is released under the Apache 2.0 License. See `LICENSE`.

## Citation

If you want to cite this repository for now, use a provisional entry like:

```bibtex
@misc{chi2026cotknot,
  title        = {Cot-Knot: Different Kinds of Knots in Chain-of-Thought},
  author       = {Yuhan Chi},
  year         = {2026},
  note         = {Workshop paper repository}
}
```

## Contact

Yuhan Chi  
Fudan University
