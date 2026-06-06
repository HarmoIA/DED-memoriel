"""
DED Mémoriel Filter — Autonomous Demo no GPU required
=====================================================

This script reports the current calibrated simulation results for the
DED Mémoriel Filter on a TruthfulQA-like benchmark size N=817.

The values are intentionally fixed to match the repository's documented
preliminary simulation results.

They do not constitute experimental validation on real logits.

Run:
    python demo_simulation.py

François Renno — Synoptisme Research Initiative, Thailand 2026
"""

import json
import os

# ── Metadata ──────────────────────────────────────────────────────────────
N = 817
SEED = 42
MODEL_ASSUMPTION = "Mistral-7B-Instruct-v0.3"
DATASET_ASSUMPTION = "TruthfulQA"
STATUS = "preliminary_simulation_not_real_logits"

# ── Calibrated preliminary simulation results ─────────────────────────────
results = {
    "meta": {
        "N": N,
        "seed": SEED,
        "model_assumption": MODEL_ASSUMPTION,
        "dataset_assumption": DATASET_ASSUMPTION,
        "status": STATUS,
        "warning": "Calibrated simulated distributions — not validated on real logits"
    },
    "baseline": {
        "covered_precision": 0.542,
        "coverage": 1.000,
        "hallucination_rate": 0.458
    },
    "DED_OFF": {
        "covered_precision": 0.779,
        "coverage": 0.548,
        "hallucination_rate": 0.121
    },
    "DED_ON": {
        "covered_precision": 0.760,
        "coverage": 0.694,
        "hallucination_rate": 0.166
    },
    "ablation": {
        "hallucination_reduction_vs_baseline_OFF": -0.736,
        "coverage_gain_cycle_memory_pp": 14.6,
        "covered_precision_change_OFF_to_ON": "77.9% -> 76.0%"
    },
    "auc_roc_entropy_signal": 0.851
}


def pct(x):
    """Format decimal as percentage with one decimal."""
    return f"{x * 100:.1f}%"


print("=" * 68)
print("  DED Mémoriel Filter — Calibrated Simulation Demo")
print(f"  Dataset assumption: {DATASET_ASSUMPTION}, N={N}")
print(f"  Model assumption:   {MODEL_ASSUMPTION}")
print(f"  Seed:               {SEED}")
print("=" * 68)

print("\n── Results ─────────────────────────────────────────────────")
print(
    "  Baseline | "
    f"Coverage: {pct(results['baseline']['coverage'])} | "
    f"Covered precision: {pct(results['baseline']['covered_precision'])} | "
    f"Hallucination: {pct(results['baseline']['hallucination_rate'])}"
)
print(
    "  DED OFF  | "
    f"Coverage: {pct(results['DED_OFF']['coverage'])}  | "
    f"Covered precision: {pct(results['DED_OFF']['covered_precision'])} | "
    f"Hallucination: {pct(results['DED_OFF']['hallucination_rate'])}"
)
print(
    "  DED ON   | "
    f"Coverage: {pct(results['DED_ON']['coverage'])}  | "
    f"Covered precision: {pct(results['DED_ON']['covered_precision'])} | "
    f"Hallucination: {pct(results['DED_ON']['hallucination_rate'])}"
)
print(f"  AUC-ROC  | {results['auc_roc_entropy_signal']:.3f}")
print("────────────────────────────────────────────────────────────")

print("\n── Ablation ────────────────────────────────────────────────")
print("  Entropy routing alone: −73.6% hallucinations vs baseline")
print("  CycleMemory: +14.6 pp coverage, from 54.8% to 69.4%")
print("  Covered precision: 77.9% → 76.0%")
print("────────────────────────────────────────────────────────────")

print("\n⚠️  WARNING")
print("   These values are based on calibrated simulated distributions.")
print("   They are not validated on real logits.")
print("   Real-logit validation on Mistral-7B-Instruct-v0.3 is planned.\n")

# ── Save simulation_results.json ─────────────────────────────────────────
with open("simulation_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("✅ simulation_results.json saved.")
print("=" * 68)

