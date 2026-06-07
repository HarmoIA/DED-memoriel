# DED Mémoriel Filter

**Selective inference without fine-tuning.**

DED Mémoriel Filter is a lightweight software prototype for entropy-based selective inference in large language models. It transforms uncertainty signals derived from logits into structured decisions to **generate**, **abstain**, or **generate cautiously**.

This release is an **alpha proof of concept**. Current results are based on controlled tests and simulated distributions. They are intended to validate the internal behavior of the prototype and motivate real-logit validation, not to establish final empirical performance.

![License: AGPLv3](https://img.shields.io/badge/License-AGPLv3-blue.svg)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Fine-tuning](https://img.shields.io/badge/Fine--tuning-No-orange.svg)
![Status](https://img.shields.io/badge/Status-Preliminary-yellow.svg)

---

## Overview

Standard language models generally generate an answer even when their internal uncertainty is high. DED Mémoriel Filter explores a simple alternative:

> when the uncertainty signal is too ambiguous, the system should abstain instead of forcing a response.

The method uses normalized Shannon entropy over top-k token probabilities as a lightweight uncertainty proxy. It does not modify model weights, does not require fine-tuning, and does not depend on external retrieval.

The project is designed as a wrapper around model logits rather than as a new language model.

---

## Status

Results in this repository are **preliminary**.

They currently come from:

- functional tests on random logits;
- structured tests on controlled probability distributions;
- simulated behavioral validation;
- calibrated simulated distributions inspired by TruthfulQA-like correctness settings.

They do **not** yet constitute experimental validation on real LLM logits.

Reference simulation setting:

- Model reference: `Mistral-7B-Instruct-v0.3`
- Dataset reference: `TruthfulQA`
- Sample size: `N = 817`
- Seed: `42`

Real-logit GPU calibration is the next validation step.

In this release, the term **“hallucination”** denotes an incorrect generated answer under a TruthfulQA-inspired binary correctness setting. It should be interpreted as a **proxy measure**, not as a full hallucination taxonomy.

---

## Core idea

DED introduces a three-phase inference cycle:

| Phase | Condition | Action |
|:---|:---|:---|
| D — Differentiated | Low entropy: `H < θ_D` | Generate normally |
| É — Equalized | High entropy: `H > θ_É` or bimodal signal | Suspend / abstain |
| R — Re-differentiated | Intermediate entropy + active memory | Generate cautiously |

Where `H` is normalized Shannon entropy computed over top-k token probabilities.

The goal is not to make the model intrinsically more truthful, but to reduce unsafe or low-confidence generations by routing uncertain cases toward abstention.

---

## Routing logic

The simplified routing rule is:

```text
if bimodal_conflict:
    phase = É
elif H > θ_É:
    phase = É
elif H < θ_D:
    phase = D
elif θ_D <= H <= θ_É and cycle_memory_active:
    phase = R
else:
    phase = É
```

Default parameters:

| Parameter | Default value | Role |
|:---|---:|:---|
| `θ_D` | `0.38` | Low-entropy threshold for Phase D |
| `θ_É` | `0.74` | High-entropy threshold for Phase É |
| `top_k` | `50` | Number of top tokens used for entropy |
| `bimodal_gap` | `0.15` | Maximum gap between top-1 and top-2 for bimodality |
| `bimodal_min_prob` | `0.25` | Minimum probability of top-2 for bimodality |

---

## Design principles

- **No fine-tuning**
- **No model weight modification**
- **No external retrieval requirement**
- **Low computational overhead**
- **Transparent abstention mechanism**
- **Compatible with models exposing logits or token probabilities**

---

## CycleMemory

`CycleMemory` is a lightweight session-level state used to track recent phase decisions and entropy traces.

It is not a vector database, does not store external documents, and does not modify model weights. Its purpose in this prototype is to support the intermediate `R` phase by allowing cautious generation when prior cycle information is available.

The effect of `CycleMemory` remains preliminary and must be validated on real logits.

---

## Validation status

Three preliminary validation layers are currently available.

| Validation layer | Description | Result |
|:---|:---|:---|
| Random-logit functional test | Verifies that raw random logits are accepted, converted to probabilities, routed through entropy computation, and suspended under high entropy. | Passed |
| Structured distribution tests | Validates the five main routing branches: peaked, diffuse, intermediate without memory, intermediate with memory, and bimodal. | 5/5 passed |
| Simulated behavioral validation | Simulates 110 cases across factual, risky myth, ambiguous, and ethical/paradoxical categories. | Passed in simulation |

These tests validate the internal logic and simulated behavior of the DED prototype. They do **not** yet establish empirical performance on real LLM logits.

---

## Functional test on random logits

A first low-level test was conducted on a random logit vector of dimension 32,000:

```python
logits = np.random.randn(32000)
```

With `seed = 42`, the detector returned:

```text
phase = Phase.E
H = 0.9821914460058359
suspension_reason = ENTROPIE_HAUTE
```

This confirms that the detector:

- accepts raw logits;
- converts logits to probabilities;
- extracts the top-k tokens;
- computes normalized Shannon entropy;
- applies the high-entropy suspension rule;
- returns usable metadata.

Ten repeated runs on random logits all produced Phase É with high entropy, which is expected for diffuse random distributions.

---

## Structured functional validation

A structured validation was conducted on five controlled probability distributions.

| Scenario | Expected phase | Obtained phase | Status |
|:---|:---:|:---:|:---:|
| Strongly peaked distribution | D | D | Passed |
| Diffuse distribution | É | É | Passed |
| Intermediate distribution without memory | É | É | Passed |
| Intermediate distribution with active memory | R | R | Passed |
| Bimodal distribution | É | É | Passed |

Result:

```text
5/5 tests passed
```

The structured validation confirms the main routing branches:

```text
low entropy → Phase D
high entropy → Phase É
intermediate entropy without memory → Phase É
intermediate entropy with memory → Phase R
bimodal conflict → Phase É
```

The bimodal case is particularly important: the system can suspend even when global entropy is not extremely high, if two competing tokens dominate the distribution.

Example:

```text
H = 0.3812
top1 = 0.4600
top2 = 0.4100
phase = É
reason = PARADOXE_STRUCTUREL
```

This validates the structural-conflict rule independently of global entropy alone.

---

## Simulated behavioral validation

An advanced simulated behavioral validation was conducted on 110 cases distributed across four categories:

| Category | Cases | Intended behavior |
|:---|---:|:---|
| `FACTUAL_SIMPLE` | 30 | Generate |
| `RISKY_MYTH` | 30 | Suspend |
| `AMBIGUOUS` | 25 | Re-differentiate when memory is active |
| `ETHICAL_PARADOX` | 25 | Suspend under structural conflict |

Global results:

| Metric | Value |
|:---|---:|
| Total cases | 110 |
| Generations | 63 |
| Suspensions | 47 |
| Coverage | 57.3% |
| Suspension rate | 42.7% |
| Mean entropy | 0.5749 |
| Simulated covered precision | 87.3% |
| Simulated hallucination proxy among generations | 12.7% |
| Useful suspension rate | 87.2% |
| False suspension rate | 12.8% |

Phase distribution:

| Phase | Count | Share |
|:---|---:|---:|
| D — Differentiated | 24 | 21.8% |
| É — Equalized | 47 | 42.7% |
| R — Re-differentiated | 39 | 35.5% |

Coverage by category:

| Category | Coverage |
|:---|---:|
| `FACTUAL_SIMPLE` | 100.0% |
| `AMBIGUOUS` | 76.0% |
| `RISKY_MYTH` | 26.7% |
| `ETHICAL_PARADOX` | 24.0% |

Interpretation:

> In simulation, DED preserves generation on simple factual cases, suspends most risky or paradoxical cases, and routes ambiguous cases toward Phase R when memory is active.

These results support the behavioral plausibility of the routing mechanism, but remain simulation-based.

---

## Preliminary simulation results — TruthfulQA-inspired setting

The following results are simulation-based only and should be interpreted as proof of concept.

The RAG and RLHF rows are contextual literature-derived reference points only. They are **not** direct same-protocol experimental baselines.

| Method | Covered precision ↑ | Coverage ↑ | Hallucination proxy ↓ | Fine-tuning | Relative cost |
|:---|---:|---:|---:|:---:|---:|
| Baseline | 54.2% | 100.0% | 45.8% | — | 100% |
| RAG* | 73.0% | ~90% | 27.0% | No | ~320% |
| RLHF* | 71.0% | ~95% | 29.0% | Yes | ~850% |
| DED v1 — OFF | 77.9% | 54.8% | 12.1% | No | ~105% |
| DED v2 — ON | 76.0% | 69.4% | 16.6% | No | ~105% |

\* RAG and RLHF values are literature-derived reference points, not same-protocol baselines.

**Covered precision** is defined as:

```text
covered_precision = n_correct / n_generated
```

**Coverage** is defined as:

```text
coverage = n_generated / n_total
```

The hallucination proxy is defined as the proportion of generated answers that are incorrect under the simulated TruthfulQA-inspired binary correctness setting.

---

## Ablation summary

| Component | Observed effect |
|:---|:---|
| Entropy routing alone — DED OFF | Reduces hallucination proxy from 45.8% to 12.1% |
| CycleMemory — DED ON | Increases coverage from 54.8% to 69.4% |
| Covered precision change | 77.9% → 76.0% |
| AUC-ROC of entropy signal | 0.851 |

With the rounded values shown above, the relative reduction from baseline to DED OFF is approximately:

```text
(45.8 - 12.1) / 45.8 = 73.6%
```

Therefore, under the simulated setting, DED OFF shows:

```text
−73.6% hallucination proxy rate vs baseline
```

Interpretation:

> DED OFF is more conservative and minimizes incorrect generations by abstaining more often.  
> DED ON recovers coverage through CycleMemory, with a modest decrease in covered precision and a higher hallucination proxy rate than OFF.

---

## Quick start

No GPU is required to run the simulation demo.

```bash
git clone https://github.com/HarmoIA/DED-memoriel
cd DED-memoriel
pip install -r requirements.txt
python demo_simulation.py
```

Minimal dependencies:

```bash
pip install numpy scikit-learn
```

---

## Minimal usage with logits or probabilities

```python
from DED_core import DEDPhaseDetector, CycleMemory, Phase

detector = DEDPhaseDetector()
memory = CycleMemory()

phase, H, meta = detector.detect(
    probs,
    has_cycle_memory=memory.is_active
)

if phase == Phase.E:
    response = "[ABSTAINED]"
else:
    response = your_generate_fn()

memory.record(
    query=query,
    phase=phase,
    entropy=H,
    was_generated=(phase != Phase.E)
)
```

`probs` may represent token probabilities derived from model logits, typically over a top-k token subset. Depending on the implementation path, raw logits may also be accepted and internally normalized.

---

## Repository structure

```text
DED-memoriel/
├── DED_core.py
├── demo_simulation.py
├── simulation_results.json
├── TECHNICAL_FAQ.md
├── VALIDATION_PLAN.md
├── CITATION.cff
├── LICENSE
├── NOTICE
└── README.md
```

Recommended future structure:

```text
DED-memoriel/
├── reports/
│   ├── 2026-06-06_functional_random_logits_test.md
│   ├── 2026-06-06_structured_functional_validation.md
│   └── 2026-06-06_simulated_behavioral_validation.md
```

---

## Technical documents

- [Technical FAQ](TECHNICAL_FAQ.md) — reviewer-oriented discussion of assumptions, simulation limits, entropy routing, and implementation details.
- [Validation Plan](VALIDATION_PLAN.md) — planned real-logit evaluation, ablations, baselines, metrics, and success criteria.

---

## Known limitations

- Current results are based on simulated distributions, not real model logits.
- Thresholds are not yet calibrated on a real-logit ROC curve.
- Only one model reference is currently used: `Mistral-7B-Instruct-v0.3`.
- Only one dataset reference is currently used: `TruthfulQA`.
- The gap between simulated and real logit distributions may be significant.
- Results may degrade under real inference conditions.
- Statistical significance has not yet been tested.
- RAG and RLHF values are contextual references, not same-protocol baselines.
- CycleMemory gain remains to be validated on real logits.
- Entropy alone may be an incomplete uncertainty signal.
- Some real hallucinations may occur with low entropy and high apparent model confidence.
- Current behavioral validation uses simulated outcomes, not human annotations.

---

## Validation roadmap

- [x] Core implementation: `DEDPhaseDetector` + `CycleMemory`
- [x] Functional test on random logits
- [x] Structured tests of the five routing branches
- [x] Simulated behavioral validation on 110 cases
- [x] Simulation with `N = 817`
- [x] OFF / ON ablation
- [x] Technical FAQ
- [x] Validation plan
- [ ] Convert validation reports to Markdown and add them under `reports/`
- [ ] GPU calibration on real Mistral-7B logits
- [ ] Real-logit validation on `Phi-3-mini-4k-instruct`
- [ ] Risk-coverage curves
- [ ] Selective risk analysis
- [ ] Calibration curves
- [ ] Statistical validation: bootstrap confidence intervals / McNemar test
- [ ] Same-protocol baseline comparison
- [ ] Multi-model validation
- [ ] Workshop paper submission

---

## Intended use

This project is intended for:

- research on selective generation and abstention;
- uncertainty-aware LLM wrappers;
- hallucination mitigation studies;
- lightweight safety mechanisms for LLM deployments;
- reproducible evaluation of entropy-based routing;
- exploration of risk-coverage trade-offs in generative models.

It should not be used as a validated safety guarantee in high-stakes settings without further empirical validation.

---

## Citation

If you use this repository, please cite:

```bibtex
@misc{renno2026ded,
  author    = {Renno, François},
  title     = {DED Mémoriel Filter: Selective Inference Without Fine-Tuning},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/HarmoIA/DED-memoriel},
  note      = {Alpha software release; functional validation and preliminary simulation on TruthfulQA-inspired setting, N=817; licensed under GNU AGPLv3}
}
```

A Zenodo DOI will be added after archival.

---

## License

From version `v0.1.0-alpha` onward, **DED Mémoriel Filter** is licensed under the **GNU Affero General Public License v3.0**.

This license was chosen to preserve openness for modified versions, including networked and API-based deployments.

Earlier public commits of this repository were released under the MIT License. From `v0.1.0-alpha` onward, the project is licensed under AGPLv3.

See the [`LICENSE`](LICENSE) file for the full license text.

---

## Copyright

Copyright (C) 2026 François Reynaud / François Renno.

**François Renno**  
Synoptisme Research Initiative  
Thailand, 2026  

Independent researcher — open to critical review.
