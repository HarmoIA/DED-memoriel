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

**In Phase R — intermediate entropy: θ_D ≤ H < θ_E**

The memory does not alter the LLM's internal generation.  
It acts as a **routing signal** at the wrapper level:

```text
if memory.is_active and memory.abstention_rate > threshold:
    → promote to Phase D generate with confidence flag
else:
    → generate with [UNCERTAIN] prefix tag
```

The rationale: if the model has been consistently uncertain across the session, a single intermediate-entropy query is more likely to be genuinely uncertain than a statistical fluctuation.

Memory provides **session-level context** that a single-query entropy reading cannot capture.

> ⚠️ This mechanism is implemented at wrapper level only.  
> It does not constitute prompt injection, retrieval augmentation, fine-tuning, or logit manipulation.

---

## 2. How were the simulated distributions calibrated?

The simulation was **not** calibrated to reproduce target results.  
It was constructed from first principles, then results were observed.

**Construction protocol:**

| Parameter | Value | Justification |
|:---|:---:|:---|
| Confident cluster mean | 0.25 | Typical low-entropy logit profile |
| Uncertain cluster mean | 0.58 | High-entropy regime |
| P(correct \| low H) | 0.80 | Conservative estimate from TruthfulQA-style behavior |
| P(hallucination \| high H) | 0.55 | Conservative uncertainty/hallucination estimate |
| Mixture ratio | 55/45 | Approximate confident/uncertain split |

Parameters were set **before** running the evaluation loop.  
Seed is fixed at `42` and reported for reproducibility.

**Known limitation:**  
The gap between simulated and real logit distributions may be significant. The AUC-ROC of `0.851` reflects discriminability within the simulation only. It is **not** a claim about real logits.

Real-logit validation is planned on `Mistral-7B-Instruct-v0.3` using TruthfulQA `N=817`.

---

## 3. Which RAG and RLHF implementations were used for comparison?

The RAG and RLHF figures are **literature-derived reference points**, not direct experimental comparisons on the same setup.

| Method | Source | Notes |
|:---|:---|:---|
| RAG | Lewis et al. 2020; Guu et al. 2020 | Literature reference point |
| RLHF | Ouyang et al. 2022 — InstructGPT | Literature reference point |
| Baseline | Mistral-7B zero-shot simulation | Fixed seed, same N=817 |

**This is a known limitation.**  
A direct comparison on identical queries with identical models is required for a rigorous claim.

The README results are intended to describe the current DED simulation, not to assert superiority over specific RAG or RLHF implementations.

> A controlled comparison using the same model, same questions, and same evaluation protocol is planned as part of the real-logit validation phase.

---

## 4. What are the current preliminary results?

Current results are based on calibrated simulated distributions using TruthfulQA `N=817`.

| Method | Covered precision | Coverage | Hallucination |
|:---|:---:|:---:|:---:|
| Baseline | 54.2% | 100% | 45.8% |
| DED v1 OFF | 77.9% | 54.8% | 12.1% |
| DED v2 ON | 76.0% | 69.4% | 16.6% |

**Ablation summary:**

- Entropy routing alone: **−73.6% hallucinations** vs baseline
- Cycle memory: **+14.6 pp coverage** from 54.8% to 69.4%
- Covered precision change: **77.9% → 76.0%**
- AUC-ROC entropy signal: **0.851**

---

## 5. Why does hallucination increase from DED OFF to DED ON?

DED ON increases coverage from `54.8%` to `69.4%`.

This means the system answers more cases instead of abstaining.  
As a result, hallucination rate rises from `12.1%` to `16.6%`.

This is an expected trade-off:

- DED OFF is more conservative.
- DED ON answers more often.
- DED ON keeps hallucination far below the baseline while improving coverage.

The purpose of CycleMemory is therefore not to minimize hallucination at all costs, but to recover useful coverage while maintaining a strong reduction compared to baseline.

---

## Summary of open validation items

| Claim | Current status | Planned validation |
|:---|:---:|:---|
| −73.6% hallucinations vs baseline | Simulation ⚠️ | Real logits |
| AUC-ROC 0.851 | Simulation ⚠️ | Real ROC curve |
| Covered precision 77.9% OFF / 76.0% ON | Simulation ⚠️ | Real logits |
| Coverage gain +14.6 pp | Simulation ⚠️ | Real-logit ablation |
| RAG/RLHF comparison | Literature reference ⚠️ | Controlled experiment |
| CycleMemory Phase R gain | Simulation ⚠️ | Ablation on real logits |

---

## Current limitation statement

The current repository demonstrates a reproducible simulation of DED-style selective inference.

It does **not** yet claim:

- experimental validation on real logits;
- superiority over RAG;
- superiority over RLHF;
- production readiness;
- clinical, legal, or safety-critical reliability.

The next required step is validation using real model logits from `Mistral-7B-Instruct-v0.3` on TruthfulQA.

---

*François Renno — Synoptisme Research Initiative, Thailand 2026*  
*Independent researcher — open to critical review.*
