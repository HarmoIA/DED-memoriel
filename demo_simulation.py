"""
DED Mémoriel Filter — Autonomous Demo (no GPU required)
========================================================
Simulates entropy-based phase detection and cycle memory
on a synthetic TruthfulQA-like dataset (N=817).

Run:
    pip install numpy
    python demo_simulation.py

François Renno — Synoptisme Research Initiative, Thailand 2026
"""

import numpy as np
import json
import os

# ── Reproducibility ────────────────────────────────────────────────────────
SEED = 42
rng  = np.random.default_rng(SEED)

# ── Parameters ─────────────────────────────────────────────────────────────
N          = 817
THETA_D    = 0.38    # Certainty threshold  → Phase D (generate)
THETA_E    = 0.468   # Suspension threshold → Phase É (abstain)
P_CORRECT  = 0.80    # P(correct | low entropy)
P_HALLUC   = 0.55    # P(hallucination | high entropy)

print("=" * 60)
print("  DED Mémoriel Filter — Simulation Demo")
print(f"  N={N}  θ_D={THETA_D}  θ_É={THETA_E}  seed={SEED}")
print("=" * 60)

# ── Simulate entropy distribution ──────────────────────────────────────────
# Mix of two Gaussians: confident (low H) + uncertain (high H)
n_confident  = int(N * 0.55)
n_uncertain  = N - n_confident

H_confident  = rng.normal(loc=0.25, scale=0.07, size=n_confident).clip(0, 1)
H_uncertain  = rng.normal(loc=0.58, scale=0.12, size=n_uncertain).clip(0, 1)
H_all        = np.concatenate([H_confident, H_uncertain])
rng.shuffle(H_all)

# ── Ground truth (binary: 1=correct, 0=hallucination) ─────────────────────
labels = np.where(
    H_all < THETA_D,
    rng.binomial(1, P_CORRECT,  N),
    rng.binomial(1, 1-P_HALLUC, N)
)

# ─────────────────────────────────────────────────────────────────────────
# BASELINE — always generate
# ─────────────────────────────────────────────────────────────────────────
baseline_correct    = labels.sum()
baseline_precision  = baseline_correct / N
baseline_halluc     = 1 - baseline_precision

# ─────────────────────────────────────────────────────────────────────────
# DED OFF — entropy routing only (no cycle memory)
# ─────────────────────────────────────────────────────────────────────────
ded_off_mask      = H_all < THETA_E          # True → generate
ded_off_generated = ded_off_mask.sum()
ded_off_correct   = labels[ded_off_mask].sum()
ded_off_coverage  = ded_off_generated / N
ded_off_precision = ded_off_correct / ded_off_generated if ded_off_generated > 0 else 0
ded_off_halluc    = 1 - ded_off_precision

# ─────────────────────────────────────────────────────────────────────────
# DED ON — entropy routing + cycle memory
# Phase R: intermediate entropy (θ_D ≤ H < θ_É) → generate with memory boost
# ─────────────────────────────────────────────────────────────────────────
phase_D = H_all < THETA_D
phase_E = H_all >= THETA_E
phase_R = ~phase_D & ~phase_E   # intermediate

# Memory boost: +10% correct rate in phase R
labels_on = labels.copy()
phase_R_idx = np.where(phase_R)[0]
for i in phase_R_idx:
    if labels_on[i] == 0 and rng.random() < 0.10:
        labels_on[i] = 1

ded_on_mask      = ~phase_E                  # D + R → generate
ded_on_generated = ded_on_mask.sum()
ded_on_correct   = labels_on[ded_on_mask].sum()
ded_on_coverage  = ded_on_generated / N
ded_on_precision = ded_on_correct / ded_on_generated if ded_on_generated > 0 else 0
ded_on_halluc    = 1 - ded_on_precision

# ─────────────────────────────────────────────────────────────────────────
# AUC-ROC (entropy signal vs ground truth)
# ─────────────────────────────────────────────────────────────────────────
def compute_auc_roc(scores, labels):
    """Trapezoidal AUC-ROC."""
    thresholds = np.sort(np.unique(scores))[::-1]
    tpr_list, fpr_list = [0.0], [0.0]
    P = labels.sum()
    N_neg = len(labels) - P
    for t in thresholds:
        pred = (scores >= t).astype(int)
        tp = ((pred == 1) & (labels == 1)).sum()
        fp = ((pred == 1) & (labels == 0)).sum()
        tpr_list.append(tp / P     if P     > 0 else 0)
        fpr_list.append(fp / N_neg if N_neg > 0 else 0)
    tpr_list.append(1.0); fpr_list.append(1.0)
    return float(np.trapz(tpr_list, fpr_list))

# Invert entropy: high entropy → likely hallucination (label=0)
auc = compute_auc_roc(1 - H_all, labels)

# ─────────────────────────────────────────────────────────────────────────
# Print results
# ─────────────────────────────────────────────────────────────────────────
print("\n── Results ──────────────────────────────────────────────")
print(f"  Baseline   | Cov: 100.0% | Prec: {baseline_precision:.1%} | Halluc: {baseline_halluc:.1%}")
print(f"  DED OFF    | Cov: {ded_off_coverage:.1%}  | Prec: {ded_off_precision:.1%} | Halluc: {ded_off_halluc:.1%}")
print(f"  DED ON     | Cov: {ded_on_coverage:.1%}  | Prec: {ded_on_precision:.1%} | Halluc: {ded_on_halluc:.1%}")
print(f"  AUC-ROC    | {auc:.3f}")
print("─────────────────────────────────────────────────────────")
print("\n⚠️  WARNING: Simulated distributions — not real logits.")
print("   GPU calibration on Mistral-7B-Instruct-v0.3 in progress.\n")

# ─────────────────────────────────────────────────────────────────────────
# Save results/simulation_results.json
# ─────────────────────────────────────────────────────────────────────────
results = {
    "meta": {
        "N": N,
        "seed": SEED,
        "theta_D": THETA_D,
        "theta_E": THETA_E,
        "warning": "Simulated distributions — not validated on real logits"
    },
    "baseline": {
        "covered_precision": round(baseline_precision, 4),
        "coverage": 1.0,
        "hallucination_rate": round(baseline_halluc, 4)
    },
    "DED_OFF": {
        "covered_precision": round(ded_off_precision, 4),
        "coverage": round(ded_off_coverage, 4),
        "hallucination_rate": round(ded_off_halluc, 4)
    },
    "DED_ON": {
        "covered_precision": round(ded_on_precision, 4),
        "coverage": round(ded_on_coverage, 4),
        "hallucination_rate": round(ded_on_halluc, 4)
    },
    "auc_roc": round(auc, 4)
}

os.makedirs("results", exist_ok=True)
with open("results/simulation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("✅ results/simulation_results.json saved.")
print("=" * 60)