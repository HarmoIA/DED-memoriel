# Validation Plan

## Current status

The current results are simulation-based and calibrated from approximate Mistral-7B-Instruct-v0.3 entropy regimes.

They are not yet validated on real logits.

## Phase 1 — Real logits validation

- Model: Mistral-7B-Instruct-v0.3
- Dataset: TruthfulQA, N = 817
- Access required: logits before sampling
- Metrics:
  - AUC-ROC
  - risk-coverage curve
  - covered accuracy
  - abstention rate
  - hallucination rate

## Phase 2 — Ablation

Conditions:

1. baseline LLM
2. entropy threshold only
3. DED without CycleMemory
4. DED with CycleMemory
5. CycleMemory randomized
6. CycleMemory reset every N samples

## Phase 3 — Generalization

Datasets:

- HaluEval
- FEVER
- FActScore
- Natural Questions
- TruthfulQA generation mode

## Phase 4 — Baselines

Compare against:

- temperature scaling
- selective prediction
- semantic uncertainty
- self-consistency
- RAG wrapper, same model
- DoLa / ITI if feasible

## Success criteria

The memory component is considered validated only if:

- DED with memory improves over entropy thresholding;
- the gain remains significant on real logits;
- the improvement is not explained only by higher abstention;
- results persist across at least two datasets or two models.
