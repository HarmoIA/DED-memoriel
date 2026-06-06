# DED Mémoriel Filter

> **Selective inference without fine-tuning.**  
> A lightweight wrapper that transforms entropy signals from LLM logits into structured abstention decisions.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![No fine-tuning](https://img.shields.io/badge/fine--tuning-none-green.svg)]()
[![Status: Preliminary](https://img.shields.io/badge/status-preliminary%20simulation-orange.svg)]()

---

## ⚠️ Status

> Results below are **preliminary** — derived from calibrated simulated distributions  
> (Mistral-7B-Instruct-v0.3, TruthfulQA N=817).  
> They do **not** yet constitute experimental validation on real logits.  
> GPU calibration is in progress.

---

## What it does

Standard LLMs always generate. They have no structured mechanism to say  
*"I am not confident enough to answer"* — and to make that decision traceable and recoverable.

DED introduces a **three-phase inference cycle**:

| Phase | Condition | Action |
|:---:|:---|:---|
| **D** — Differentiated | Low entropy H < θ_D | Generate normally |
| **É** — Equalized | High entropy H > θ_É or bimodal | Suspend — do not hallucinate |
| **R** — Re-differentiated | Intermediate H + active memory | Generate enriched by cycle memory |

The signal is **Shannon entropy normalized over top-k tokens** — extracted directly from logits.  
No weights modified. No fine-tuning. No external retrieval.

---

## Preliminary results (simulation, N=817)

| Method | Covered precision | Coverage | Hallucination | Fine-tuning | Relative cost |
|:---|:---:|:---:|:---:|:---:|:---:|
| Baseline | 54.2% | 100% | 45.8% | — | 100% |
| RAG | 73.0% | ~90% | 27.0% | No | ~320% |
| RLHF | 71.0% | ~95% | 29.0% | Yes | ~850% |
| **DED v1 (OFF)** | **77.9%** | 54.8% | **12.1%** | **No** | **~105%** |
| **DED v2 (ON)** | **76.0%** | 69.4% | 16.6% | **No** | **~105%** |

> **Covered precision** = n_correct / n_generated (Geifman & El-Yaniv, 2017).  
> Comparisons with RAG/RLHF are on covered precision only.

**Ablation (OFF vs ON):**
- Entropy routing alone: **−76.5% hallucinations** vs baseline
- Cycle memory: **+14.6 pp coverage** at near-stable covered precision (79.9% → 79.1%)
- AUC-ROC (entropy signal): **0.851**

---

## Quick start (no GPU required)

```bash
git clone https://github.com/HarmoIA/DED-memoriel
cd DED-memoriel
pip install numpy
python demo_simulation.py
```

---

## Minimal usage (real logits)

```python
from DED_core import DEDPhaseDetector, CycleMemory, Phase

detector = DEDPhaseDetector()   # recalibrate thresholds on real logits
memory   = CycleMemory()

phase, H, meta = detector.detect(probs, has_cycle_memory=memory.is_active)

if phase == Phase.E:
    response = "[ABSTAINED]"
else:
    response = your_generate_fn()

memory.record(query=query, phase=phase, entropy=H, was_generated=(phase != Phase.E))
```

---

## Repository structure

```
DED-memoriel/
├── DED_core.py                        # Core — DEDPhaseDetector + CycleMemory
├── demo_simulation.py                 # Autonomous demo — no GPU needed
├── results/
│   └── simulation_results.json        # Verified results N=817
├── CITATION.cff
└── README.md
```

---

## Known limitations

- Thresholds calibrated on simulation — not yet on real logit ROC curve
- Single model (Mistral-7B), single dataset (TruthfulQA)
- Gap between simulated and real distributions may be significant
- Results may degrade in real conditions
- Statistical significance not yet tested (bootstrap / McNemar pending)

---

## Roadmap

- [x] Core implementation (DEDPhaseDetector + CycleMemory)
- [x] Simulation N=817 — verified results
- [x] Ablation OFF/ON
- [ ] GPU calibration on real Mistral-7B logits (A100 — in progress)
- [ ] Risk–coverage curves
- [ ] Statistical validation (bootstrap / McNemar)
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
  note      = {Preliminary results — simulation on TruthfulQA N=817}
}
```

---

François Renno — Synoptisme Research Initiative, Thailande 2026  
*Independent researcher — open to critical review.*
