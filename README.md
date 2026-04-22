# Cot-Knot

`Cot-Knot` is a workshop-stage research repo for one central claim:

> self-correction in chain-of-thought is **not** a universal quality signal, because different domains contain different kinds of local reasoning knots.

This repository is the lightweight public-facing home for the **knot** line of the broader `SVDomain` project.

## Core idea

The project studies when a model's apparent “self-correction” actually means something reliable.

In math, local breaks in reasoning can still be compatible with successful rebinding and recovery.
In coding, superficially similar corrections often correspond to a different object: unstable execution-state serialization, where the text no longer faithfully tracks the latent program state.

The working hypothesis is:

- **math knots** are often explicit and text-visible
- **coding knots** are often execution-semantic and only partially visible in prose
- therefore the same surface pattern of “correction” should not be assumed to carry the same semantics across domains

## Current paper direction

This repo tracks the workshop-paper direction currently developed in the private `SVDomain/workshop` drafts, especially the `paper_v6` line:

- **domain-conditioned CoT quality signals**
- **different kinds of knots in math vs coding**
- **local state breaks as mechanism evidence**
- **warning signs under RL posttraining / score drift**

In the current framing, the paper makes three empirical points:

1. Compact CoT-based quality probes work strongly in math, reasonably in science, and poorly in coding.
2. Math and coding exhibit **different knot types**, rather than one universal self-correction phenomenon.
3. RL posttraining can preserve ranking signal while breaking the original binary operating point of an SFT-era verifier.

## What “knot” means here

“Knot” is used as a mechanism-level term for a local failure or entanglement in reasoning state.

Examples include:

- explicit contradiction
- incompatible redefinition
- leaked case frame
- unresolved repair
- lost execution state
- patchy backtracking
- branch / loop / index-state inconsistency

The key claim is not merely that knots occur, but that **their meaning depends on domain**.

## Relationship to `SVDomain`

This repository is conceptually extracted from:

- `SVDomain/workshop/`
- `SVDomain/docs/`
- `SVDomain/results/`

`SVDomain` remains the larger research workspace with experiments, artifacts, and draft iterations.
`Cot-Knot` is intended to become the smaller repo centered on the workshop paper, knot framing, and minimal reproducible evidence.

## Planned contents

As the repo is cleaned and opened up, it will gradually contain:

- the workshop paper draft
- a concise project overview
- figure assets for knot/domain comparisons
- annotation guidelines for local state breaks
- small tables or summaries needed to support the workshop claims

## Status

This is an early public scaffold.

The current version is intentionally minimal and serves as the initial anchor for the `Cot-Knot` workshop repo while the paper and supporting materials are being separated from the larger internal workspace.

## Contact

Yuhan Chi  
Fudan University

