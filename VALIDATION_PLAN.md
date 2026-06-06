# Validation Plan

## Current status

The current results are based on calibrated simulated distributions using `Mistral-7B-Instruct-v0.3` assumptions and TruthfulQA `N=817`.

They are **not yet validated on real logits**.

Current preliminary simulation results:

| Method | Covered precision | Coverage | Hallucination |
|:---|:---:|:---:|:---:|
| Baseline | 54.2% | 100% | 45.8% |
| DED v1 OFF | 77.9% | 54.8% | 12.1% |
| DED v2 ON | 76.0% | 69.4% | 16.6% |

Current ablation summary:

- Entropy routing alone: **−73.6% hallucinations** vs baseline
- CycleMemory: **+14.6 pp coverage** from 54.8% to 69.4%
- Covered precision change: **77.9% → 76.0%**
- AUC-ROC entropy signal: **0.851**

---

## Phase 1 — Real logits validation

Goal: reproduce the simulation protocol using real model logits.

- Model: `Mistral-7B-Instruct-v0.3`
- Dataset: TruthfulQA
- Sample size: `N=817`
- Access required: logits before sampling
- Inference mode: zero-shot or fixed prompt template
- Seed: fixed and reported
- Hardware target: GPU environment with sufficient VRAM

Metrics:

- AUC-ROC
- risk-coverage curve
- covered precision
- coverage
- abstention rate
- hallucination rate
- calibration curve
- confidence / entropy distribution

Expected output files:

- `real_logits_results.json`
- `risk_coverage_curve.png`
- `roc_curve.png`
- `entropy_distribution.png`

---

## Phase 2 — Ablation study

Goal: isolate the effect of entropy routing and CycleMemory.

Conditions:

1. Baseline LLM without DED wrapper
2. Entropy threshold only
3. DED without CycleMemory
4. DED with CycleMemory
5. CycleMemory randomized
6. CycleMemory reset every N samples
7. Random abstention matched for coverage
8. Oracle threshold upper bound

Required comparisons:

| Comparison | Purpose |
|:---|:---|
| Baseline vs DED OFF | Measure entropy-routing effect |
| DED OFF vs DED ON | Measure CycleMemory effect |
| DED ON vs randomized memory | Check if memory signal is meaningful |
| DED ON vs coverage-matched random abstention | Check if gain is not only due to abstention |
| DED ON vs oracle threshold | Estimate remaining headroom |

---

## Phase 3 — Generalization

Goal: test whether the method transfers beyond the initial TruthfulQA simulation.

Datasets:

- TruthfulQA generation mode
- HaluEval
- FEVER
- FActScore
- Natural Questions
- Open-ended QA subset if available

Models:

- `Mistral-7B-Instruct-v0.3`
- one smaller open-weight model if feasible
- one larger open-weight model if feasible

The method should be considered more robust if improvements persist across at least two datasets or two model families.

---

## Phase 4 — Baselines

Goal: compare DED against established uncertainty and hallucination-reduction methods.

Compare against:

- temperature scaling
- selective prediction
- semantic uncertainty
- self-consistency
- verbalized confidence
- RAG wrapper using the same model
- DoLa if feasible
- ITI if feasible

Important constraint:

All comparisons should use:

- the same dataset;
- the same model where possible;
- the same prompt template;
- the same scoring protocol;
- the same coverage / abstention reporting.

---

## Phase 5 — Reporting standard

Every reported result should include:

- model name and version;
- dataset and split;
- sample size;
- seed;
- prompt template;
- threshold values;
- whether logits are simulated or real;
- coverage;
- covered precision;
- hallucination rate;
- abstention rate;
- AUC-ROC when applicable;
- confidence interval or bootstrap estimate when feasible.

No result should be described as experimentally validated unless it was measured on real logits.

---

## Success criteria

The memory component is considered validated only if:

- DED with CycleMemory improves coverage over entropy-only routing;
- the gain remains significant on real logits;
- the improvement is not explained only by lower abstention;
- hallucination remains substantially below baseline;
- results persist across at least two datasets or two models.

For the current preliminary target, the real-logit validation should test whether the following simulation pattern is preserved:

| Quantity | Simulation target |
|:---|:---:|
| Hallucination reduction vs baseline | −73.6% |
| Coverage gain from CycleMemory | +14.6 pp |
| Covered precision OFF to ON | 77.9% → 76.0% |
| AUC-ROC entropy signal | 0.851 |

---

## Limitations

The present simulation does **not** establish:

- real-logit validation;
- superiority over RAG;
- superiority over RLHF;
- production readiness;
- safety-critical reliability.

The current repository should be interpreted as a reproducible preliminary simulation and implementation scaffold.

---

## Next milestone

The next milestone is to run `Mistral-7B-Instruct-v0.3` on TruthfulQA `N=817`, extract real logits, and reproduce the complete DED evaluation pipeline.

The expected deliverable is a real-logit validation report with curves, JSON metrics, and reproducible scripts.
