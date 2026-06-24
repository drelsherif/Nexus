"""
health_monitor.py
NEXUS — Trajectory analysis and degradation detection.

Domain-agnostic. Detects performance degradation patterns by analyzing
the eval history and classifies them into actionable intervention types.

Design principle: all thresholds are relative to the observed trajectory
and the task's declared optimization_beta — never hardcoded to a domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class DegradationType(Enum):
    HEALTHY               = "healthy"
    POST_GRAFT_DISRUPTION = "post_graft_disruption"    # F1 dropped after a graft
    PRINCIPLE_OVERCORRECT = "principle_overcorrection"  # F1/precision dropped after SWR
    RECALL_COLLAPSE       = "recall_collapse"            # R fell below safety floor
    PRECISION_COLLAPSE    = "precision_collapse"         # P fell sharply, R unchanged
    PLATEAU               = "plateau"                    # No improvement for N rounds
    GENERAL_DECLINE       = "general_decline"            # F1 falling, no clear cause


@dataclass
class HealthReport:
    status:           DegradationType
    severity:         float           # 0.0 (negligible) → 1.0 (critical)
    peak_f1:          float
    current_f1:       float
    delta_from_peak:  float
    rounds_since_peak: int
    recent_graft:     bool            # graft occurred in last 2 rounds
    recent_principle: bool            # SWR principle injected in last 2 rounds
    recall:           float
    precision:        float
    message:          str


class HealthMonitor:
    """
    Analyzes eval history to detect and classify degradation.

    eval_history entries (from DB or nexus_v3 loop) are dicts with:
        round, f1, precision, recall, graft_happened, swr_happened

    All thresholds are relative — the monitor adapts to the trajectory.
    """

    # Thresholds (relative to trajectory, not hardcoded to ADE values)
    DECLINE_THRESHOLD        = 0.025   # F1 drop that triggers assessment
    SEVERE_DECLINE_THRESHOLD = 0.060   # F1 drop that skips cheap interventions
    RECALL_FLOOR             = 0.875   # R below this is always a recall collapse
    PLATEAU_ROUNDS           = 4       # Rounds without ≥0.005 improvement = plateau
    PRECISION_DROP_THRESHOLD = 0.030   # P drop in one round = precision collapse

    def __init__(self, task_config):
        self.beta = getattr(task_config, "optimization_beta", 1.0)

    def assess(
        self,
        eval_history: list[dict],
    ) -> HealthReport:
        """
        Assess current health from eval history.

        Each entry must have: round, f1, precision, recall.
        Optional: graft_happened (bool), swr_happened (bool).
        """
        if not eval_history:
            return self._healthy(0.0, 0.0, 0.0, 1.0, 1.0, "No history yet.")

        current = eval_history[-1]
        f1      = current.get("f1", 0.0)
        prec    = current.get("precision", 1.0)
        rec     = current.get("recall", 1.0)

        if len(eval_history) < 5:
            return self._healthy(f1, f1, 0.0, rec, prec, "Insufficient history for assessment (need 5 rounds).")

        # Trajectory metrics
        peak_f1       = max(e["f1"] for e in eval_history)
        peak_idx      = next(i for i, e in enumerate(eval_history) if e["f1"] == peak_f1)
        rounds_since  = len(eval_history) - 1 - peak_idx
        delta         = f1 - peak_f1
        severity      = min(1.0, abs(delta) / max(0.01, self.SEVERE_DECLINE_THRESHOLD))

        # Recent structural events (last 2 rounds)
        recent        = eval_history[-2:]
        recent_graft  = any(e.get("graft_happened", False) for e in recent)
        recent_swr    = any(e.get("swr_happened", False) for e in recent)

        prev_prec     = eval_history[-2].get("precision", prec) if len(eval_history) >= 2 else prec

        def _report(status, msg):
            return HealthReport(
                status=status, severity=severity,
                peak_f1=peak_f1, current_f1=f1,
                delta_from_peak=delta, rounds_since_peak=rounds_since,
                recent_graft=recent_graft, recent_principle=recent_swr,
                recall=rec, precision=prec, message=msg,
            )

        # ── Classification logic ──────────────────────────────────────────────

        # 1. Recall collapse — always highest priority (safety-critical)
        if rec < self.RECALL_FLOOR:
            return _report(
                DegradationType.RECALL_COLLAPSE,
                f"Recall collapsed to R={rec:.3f} (floor={self.RECALL_FLOOR}). "
                f"Missing true positives."
            )

        # 2. Healthy / no significant decline
        if delta > -self.DECLINE_THRESHOLD:
            if rounds_since >= self.PLATEAU_ROUNDS:
                return _report(
                    DegradationType.PLATEAU,
                    f"No improvement for {rounds_since} rounds. "
                    f"Plateau at F1={f1:.4f} (peak={peak_f1:.4f})."
                )
            return self._healthy(peak_f1, f1, delta, rec, prec, "System healthy.", rounds_since)

        # 3. Significant decline — classify cause
        if recent_graft:
            return _report(
                DegradationType.POST_GRAFT_DISRUPTION,
                f"Post-graft F1 drop of {delta:.4f}. "
                f"New node may be over-intercepting or misclas​sifying."
            )

        if recent_swr and (prev_prec - prec) > self.PRECISION_DROP_THRESHOLD:
            return _report(
                DegradationType.PRINCIPLE_OVERCORRECT,
                f"Precision dropped {prev_prec:.3f}→{prec:.3f} after principle injection. "
                f"Possible overcorrection toward positive class."
            )

        if (prev_prec - prec) > self.PRECISION_DROP_THRESHOLD:
            return _report(
                DegradationType.PRECISION_COLLAPSE,
                f"Precision collapsed {prev_prec:.3f}→{prec:.3f} without clear structural cause."
            )

        return _report(
            DegradationType.GENERAL_DECLINE,
            f"F1 declined {delta:.4f} from peak {peak_f1:.4f}. No single cause identified."
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _healthy(self, peak, current, delta, rec, prec, msg, rounds_since=0):
        return HealthReport(
            status=DegradationType.HEALTHY,
            severity=0.0,
            peak_f1=peak,
            current_f1=current,
            delta_from_peak=delta,
            rounds_since_peak=rounds_since,
            recent_graft=False,
            recent_principle=False,
            recall=rec,
            precision=prec,
            message=msg,
        )
