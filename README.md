# DED Mémoriel Filter

**Selective inference without fine-tuning.**

A lightweight wrapper that transforms entropy signals from LLM logits into structured abstention decisions.

![License: AGPLv3](https://img.shields.io/badge/License-AGPLv3-blue.svg)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Fine-tuning](https://img.shields.io/badge/Fine--tuning-No-orange.svg)
![Status](https://img.shields.io/badge/Status-Preliminary-yellow.svg)

---

## ⚠️ Status

Results below are **preliminary** and derived from calibrated simulated distributions:

- Model reference: `Mistral-7B-Instruct-v0.3`
- Dataset reference: `TruthfulQA`
- N = 817
- Seed = 42

They do **not** yet constitute experimental validation on real logits.

Real-logit GPU calibration is the next validation step.

---

## What it does

Standard LLMs always generate. They have no structured mechanism to say:

> “I am not confident enough to answer.”

DED introduces a three-phase inference cycle:

| Phase | Condition | Action |
|:---|:---|:---|
| D — Differentiated | Low entropy: `H < θ_D` | Generate normally |
| É — Equalized | High entropy: `H > θ_É` or bimodal signal | Suspend / abstain |
| R — Re-differentiated | Intermediate entropy + active memory | Generate cautiously |

The signal is normalized Shannon entropy over top-k tokens, extracted from logits.

No weights modified.  
No fine-tuning.  
No external retrieval required.

---

## Preliminary results — simulation, N = 817

> These results are simulation-based only and should be interpreted as proof of concept.

| Method | Covered precision ↑ | Coverage ↑ | Hallucination ↓ | Fine-tuning | Relative cost |
|:---|---:|---:|---:|:---:|---:|
| Baseline | 54.2% | 100.0% | 45.8% | — | 100% |
| RAG* | 73.0% | ~90% | 27.0% | No | ~320% |
| RLHF* | 71.0% | ~95% | 29.0% | Yes | ~850% |
| DED v1 — OFF | 77.9% | 54.8% | 12.1% | No | ~105% |
| DED v2 — ON | 76.0% | 69.4% | 16.6% | No | ~105% |

\* RAG and RLHF numbers are literature-derived reference points, not direct same-protocol experimental baselines.

**Covered precision** is defined as:

```text
covered_precision = n_correct / n_generated
```

---

## Ablation summary — OFF vs ON

| Component | Observed effect |
|:---|:---|
| Entropy routing alone — DED OFF | Reduces hallucination from 45.8% to 12.1% |
| CycleMemory — DED ON | Increases coverage from 54.8% to 69.4% |
| Covered precision change | 77.9% → 76.0% |
| AUC-ROC of entropy signal | 0.851 |

With the rounded values shown above, the relative hallucination reduction from baseline to DED OFF is approximately:

```text
(45.8 - 12.1) / 45.8 = 73.6%
```

So the conservative reported value is:

```text
−73.6% hallucination rate vs baseline
```

Interpretation:

> DED OFF is more conservative and minimizes hallucination.  
> DED ON recovers coverage through CycleMemory, with a modest decrease in covered precision and a higher hallucination rate than OFF.

---

## Quick start — no GPU required

```bash
git clone https://github.com/HarmoIA/DED-memoriel
cd DED-memoriel
pip install -r requirements.txt
python demo_simulation.py
```

If `requirements.txt` is not available yet, install the minimal dependencies manually:

```bash
pip install numpy scikit-learn
```

---

## Minimal usage — real logits

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

---

## Technical documents

- [Technical FAQ](TECHNICAL_FAQ.md) — anticipated reviewer questions, simulation limits, and implementation details.
- [Validation Plan](VALIDATION_PLAN.md) — planned real-logit validation, ablations, baselines, and success criteria.

---

## Known limitations

- Thresholds are calibrated on simulated distributions, not yet on a real-logit ROC curve.
- Single model reference: Mistral-7B-Instruct-v0.3.
- Single dataset reference: TruthfulQA.
- The gap between simulated and real logit distributions may be significant.
- Results may degrade in real conditions.
- Statistical significance has not yet been tested.
- RAG and RLHF values are literature-derived reference points, not same-protocol baselines.
- CycleMemory gain remains to be validated on real logits.

---

## Roadmap

- [x] Core implementation: `DEDPhaseDetector` + `CycleMemory`
- [x] Simulation N = 817
- [x] Ablation OFF / ON
- [x] Technical FAQ
- [x] Validation plan
- [ ] GPU calibration on real Mistral-7B logits
- [ ] Risk-coverage curves
- [ ] Statistical validation: bootstrap / McNemar
- [ ] Same-protocol baselines
- [ ] Multi-model validation
- [ ] NeurIPS 2026 Workshop submission

---

## Citation

```bibtex
@misc{renno2026ded,
  author    = {Renno, François},
  title     = {DED Mémoriel Filter: Selective Inference Without Fine-Tuning},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/HarmoIA/DED-memoriel},
  note      = {Preliminary results — simulation on TruthfulQA N=817; licensed under GNU AGPLv3}
}
```

---

## License

From version `v0.1.0-alpha` onward, **DED Mémoriel Filter** is licensed under the **GNU Affero General Public License v3.0**.

This license was chosen to preserve openness for modified versions, including networked/API-based deployments.

Earlier public commits of this repository were released under the MIT License. From `v0.1.0-alpha` onward, the project is licensed under AGPLv3.

See the [`LICENSE`](LICENSE) file for the full license text.

Copyright (C) 2026 François Reynaud / François Renno.

---

**François Renno**  
Synoptisme Research Initiative  
Thailand, 2026  

Independent researcher — open to critical review.


