# Technical FAQ

> Anticipated questions from technical reviewers.  
> Last updated: June 2026 — François Renno

---

## 1. What exactly does CycleMemory do in Phase R?

`CycleMemory` is **not** a retrieval system and does **not** modify model weights.

It is a lightweight session-level state object that tracks:
- The entropy trajectory across previous turns (`H_history`)
- Whether previous queries were generated or abstained (`phase_history`)
- A running abstention rate (`abstention_rate`)

**In Phase R (intermediate entropy: θ_D ≤ H < θ_É):**

The memory does not alter the LLM's internal generation.  
It acts as a **routing signal** at the wrapper level:

```
if memory.is_active and memory.abstention_rate > threshold:
    → promote to Phase D (generate with confidence flag)
else:
    → generate with [UNCERTAIN] prefix tag
```

The rationale: if the model has been consistently uncertain across
the session, a single intermediate-entropy query is more likely to
be genuinely uncertain than a statistical fluctuation.  
Memory provides **session-level context** that a single-query
entropy reading cannot capture.

> ⚠️ This mechanism is implemented at wrapper level only.  
> It does not constitute prompt injection or logit manipulation.

---

## 2. How were the simulated distributions calibrated?

The simulation was **not** calibrated to reproduce target results.  
It was constructed from first principles, then results were observed.

**Construction protocol:**

| Parameter | Value | Justification |
|:---|:---:|:---|
| Confident cluster mean | 0.25 | Typical low-entropy logit profile (Mistral-7B) |
| Uncertain cluster mean | 0.58 | Observed high-entropy regime |
| P(correct \| low H) | 0.80 | Conservative estimate from TruthfulQA literature |
| P(hallucination \| high H) | 0.55 | Conservative — real rate likely higher |
| Mixture ratio | 55/45 | Approximate empirical split |

Parameters were set **before** running the evaluation loop.  
Seed is fixed (42) and reported for full reproducibility.

**Known limitation:**  
The gap between simulated and real logit distributions may be
significant. The AUC-ROC of 0.851 reflects discriminability
within the simulation — it is not a claim about real logits.  
GPU calibration will produce a real ROC curve on Mistral-7B-Instruct-v0.3.

---

## 3. Which RAG and RLHF implementations were used for comparison?

The RAG and RLHF figures are **literature-derived reference points**,
not direct experimental comparisons on the same setup.

| Method | Source | Notes |
|:---|:---|:---|
| RAG | Lewis et al. (2020), Guu et al. (2020) | Typical TruthfulQA performance range |
| RLHF | Ouyang et al. (2022) — InstructGPT | Reported accuracy on truthfulness benchmarks |
| Baseline | Mistral-7B zero-shot (simulated) | Fixed seed, same N=817 |

**This is a known limitation** — direct comparison on identical
queries with identical models is required for a rigorous claim.  
The table in the README is intended to situate the approach,
not to assert superiority over specific implementations.

> A controlled comparison (same model, same queries, RAG wrapper vs
> DED wrapper) is planned as part of the GPU validation phase.

---

## Summary of open validation items

| Claim | Current status | Planned validation |
|:---|:---:|:---|
| −76.5% hallucinations vs baseline | Simulation ⚠️ | Real logits — A100 |
| AUC-ROC 0.851 | Simulation ⚠️ | Real ROC curve |
| Covered precision 77.9% | Simulation ⚠️ | Real logits — A100 |
| RAG/RLHF comparison | Literature ref ⚠️ | Controlled experiment |
| CycleMemory phase R gain | Simulation ⚠️ | Ablation on real logits |

---

*François Renno — Synoptisme Research Initiative, Thailand 2026*  
*Independent researcher — open to critical review.*