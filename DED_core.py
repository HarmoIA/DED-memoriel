"""
DED_core.py — DED Mémoriel Filter
===================================
Core implementation of the DED (Differentiated-Equalized-Re-differentiated)
selective inference wrapper.

No fine-tuning. No weight modification. Logits-only.

Author  : François Renno — Synoptisme Research Initiative
Version : 0.1.0 — June 2026
Status  : Preliminary — thresholds calibrated on simulation
License : MIT
"""

from __future__ import annotations
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


# ══════════════════════════════════════════════════════════════════
#  PHASE ENUM
# ══════════════════════════════════════════════════════════════════

class Phase(Enum):
    D = "DIFFERENTIATED"      # Low entropy  → generate normally
    E = "EQUALIZED"           # High entropy → suspend, do not hallucinate
    R = "RE_DIFFERENTIATED"   # Intermediate + memory → generate enriched


# ══════════════════════════════════════════════════════════════════
#  DETECTION METADATA
# ══════════════════════════════════════════════════════════════════

@dataclass
class DetectionMeta:
    entropy:           float
    phase:             Phase
    is_bimodal:        bool
    suspension_reason: Optional[str]   # None if generated
    top_k:             int
    theta_D:           float
    theta_E:           float


# ══════════════════════════════════════════════════════════════════
#  DED PHASE DETECTOR
# ══════════════════════════════════════════════════════════════════

class DEDPhaseDetector:
    """
    Detects the inference phase (D / É / R) from a probability distribution
    over the vocabulary (last generated token).

    Parameters
    ----------
    seuil_D : float
        Below this entropy → Phase D (safe generation).
        Default: 0.38 (simulation-calibrated — recalibrate on real logits).
    seuil_E : float
        Above this entropy → Phase É (suspension).
        Default: 0.74 (conservative — Youden optimal on simulation: 0.468).
    top_k : int
        Number of top tokens used for entropy computation.
    bimodal_gap : float
        Minimum probability gap between top-2 tokens to trigger bimodal detection.
    bimodal_min_prob : float
        Minimum probability of second token to confirm bimodality.

    ⚠️  Recalibrate seuil_D and seuil_E on real logits using:
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(labels, H_real)
        seuil_E = thresholds[np.argmax(tpr - fpr)]   # Youden index
        seuil_D = np.percentile(H_real[labels==0], 30)
    """

    def __init__(
        self,
        seuil_D:          float = 0.38,
        seuil_E:          float = 0.74,
        top_k:            int   = 50,
        bimodal_gap:      float = 0.15,
        bimodal_min_prob: float = 0.25,
    ):
        self.seuil_D          = seuil_D
        self.seuil_E          = seuil_E
        self.top_k            = top_k
        self.bimodal_gap      = bimodal_gap
        self.bimodal_min_prob = bimodal_min_prob

    # ── Entropy ───────────────────────────────────────────────────

    def compute_entropy(self, probs: np.ndarray) -> float:
        """
        Normalized Shannon entropy over top-k tokens.
        Returns H ∈ [0, 1].
        """
        idx   = np.argsort(probs)[-self.top_k:]
        top_p = probs[idx]
        top_p = top_p / top_p.sum()                    # renormalize
        mask  = top_p > 1e-10
        H     = -np.sum(top_p[mask] * np.log2(top_p[mask]))
        return float(H / np.log2(self.top_k))          # normalize ∈ [0,1]

    # ── Bimodal detection ─────────────────────────────────────────

    def is_bimodal(self, probs: np.ndarray) -> bool:
        """
        Detects structural paradox: two tokens with near-equal high probability.
        Triggers Phase É regardless of entropy level.
        """
        idx    = np.argsort(probs)[-2:]
        p1, p2 = probs[idx[1]], probs[idx[0]]          # descending
        return (
            p2 >= self.bimodal_min_prob and
            (p1 - p2) <= self.bimodal_gap
        )

    # ── Main detection ────────────────────────────────────────────

    def detect(
        self,
        probs:            np.ndarray,
        has_cycle_memory: bool = False,
    ) -> tuple[Phase, float, dict]:
        """
        Determine inference phase from token probability distribution.

        Parameters
        ----------
        probs : np.ndarray
            Probability distribution over vocabulary (shape: vocab_size,).
            Must sum to ~1.0. Obtain via softmax(logits).
        has_cycle_memory : bool
            Whether cycle memory is active (enables Phase R).

        Returns
        -------
        phase : Phase
            D, É, or R
        H : float
            Normalized entropy ∈ [0, 1]
        meta : dict
            Detection metadata (bimodal, suspension_reason, thresholds)
        """
        H       = self.compute_entropy(probs)
        bimodal = self.is_bimodal(probs)

        suspension_reason = None

        # ── Structural paradox (bimodal) → always suspend ─────────
        if bimodal:
            phase             = Phase.E
            suspension_reason = "PARADOXE_STRUCTUREL"

        # ── High entropy → suspend ────────────────────────────────
        elif H > self.seuil_E:
            phase             = Phase.E
            suspension_reason = "ENTROPIE_HAUTE"

        # ── Low entropy → generate safely ─────────────────────────
        elif H < self.seuil_D:
            phase = Phase.D

        # ── Intermediate entropy ──────────────────────────────────
        else:
            if has_cycle_memory:
                phase = Phase.R   # memory enriches the response
            else:
                phase             = Phase.E
                suspension_reason = "ENTROPIE_INTERMEDIAIRE_SANS_MEMOIRE"

        meta = {
            "entropy":           H,
            "is_bimodal":        bimodal,
            "suspension_reason": suspension_reason,
            "has_cycle_memory":  has_cycle_memory,
            "theta_D":           self.seuil_D,
            "theta_E":           self.seuil_E,
        }

        return phase, H, meta


# ══════════════════════════════════════════════════════════════════
#  CYCLE MEMORY
# ══════════════════════════════════════════════════════════════════

@dataclass
class CycleRecord:
    query:             str
    phase:             Phase
    entropy:           float
    was_generated:     bool
    suspension_reason: Optional[str] = None
    enrichment:        float         = 1.0


class CycleMemory:
    """
    Accumulates inter-cycle state to enable Phase R (re-differentiation).

    The memory becomes active after the first generated response (Phase D or R).
    It tracks suspension history and computes an enrichment factor E_n
    that modulates generation in Phase R.

    Ablation note
    -------------
    To isolate the effect of cycle memory (OFF/ON ablation):
        # OFF — entropy routing only
        phase, H, meta = detector.detect(probs, has_cycle_memory=False)

        # ON — full DED with memory
        phase, H, meta = detector.detect(probs, has_cycle_memory=memory.is_active)
    """

    def __init__(self, enrichment_decay: float = 0.9):
        self.cycles:           list[CycleRecord] = []
        self.enrichment_decay: float             = enrichment_decay
        self._enrichment:      float             = 1.0

    # ── State ─────────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        """Memory is active once at least one response has been generated."""
        return any(c.was_generated for c in self.cycles)

    @property
    def enrichment_factor(self) -> float:
        """Current enrichment factor E_n ∈ [1.0, ∞)."""
        return self._enrichment

    @property
    def n_cycles(self) -> int:
        return len(self.cycles)

    # ── Record ────────────────────────────────────────────────────

    def record(
        self,
        query:             str,
        phase:             Phase,
        entropy:           float,
        was_generated:     bool,
        suspension_reason: Optional[str] = None,
    ) -> None:
        """
        Record a cycle and update enrichment factor.
        Enrichment increases with each suspension, decays with each generation.
        """
        record = CycleRecord(
            query             = query,
            phase             = phase,
            entropy           = entropy,
            was_generated     = was_generated,
            suspension_reason = suspension_reason,
            enrichment        = self._enrichment,
        )
        self.cycles.append(record)

        # Update enrichment
        if not was_generated:
            self._enrichment *= (1.0 + (1.0 - self.enrichment_decay))
        else:
            self._enrichment = max(1.0, self._enrichment * self.enrichment_decay)

    # ── Summary ───────────────────────────────────────────────────

    def summary(self) -> dict:
        """Returns a summary dict for logging and reporting."""
        n_total     = len(self.cycles)
        n_suspended = sum(1 for c in self.cycles if not c.was_generated)
        phase_counts = {"D": 0, "E": 0, "R": 0}
        for c in self.cycles:
            phase_counts[c.phase.name] += 1

        return {
            "total_cycles":       n_total,
            "n_generated":        n_total - n_suspended,
            "n_suspended":        n_suspended,
            "suspension_rate":    n_suspended / n_total if n_total > 0 else 0.0,
            "enrichment_factor":  self._enrichment,
            "phase_counts":       phase_counts,
            "memory_active":      self.is_active,
        }

    def reset(self) -> None:
        """Reset memory state (new session)."""
        self.cycles      = []
        self._enrichment = 1.0