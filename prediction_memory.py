"""
PredictionMemory — Sliding-window majority voting + context-aware smoothing.

Keeps the last N raw predictions, applies majority vote, and uses
previously confirmed words to bias future predictions (context boost).
"""

import time
from collections import Counter, deque
from typing import Optional, Tuple, List, Dict


class PredictionMemory:
    """
    Two-stage confidence engine:

    Stage 1 — Window smoothing
        Buffer the last `window` predictions and pick the majority.
        Only surface a prediction when it has appeared ≥ `min_votes` times.

    Stage 2 — Context boost
        If the majority candidate matches a word that often follows the
        previously confirmed word, bump its effective confidence by
        `context_bonus`.

    Stage 3 — Stability gate
        A prediction must be the stable majority for `stable_frames` frames
        before it is "ready to accept".
    """

    # Tiny bigram co-occurrence table (word → likely next words)
    _BIGRAMS: Dict[str, List[str]] = {
        "I":      ["WANT", "NEED", "AM", "LOVE", "FEEL"],
        "WANT":   ["WATER", "FOOD", "HELP", "MORE", "TO"],
        "NEED":   ["HELP", "WATER", "FOOD", "DOCTOR", "MORE"],
        "PLEASE": ["HELP", "STOP", "WAIT", "COME"],
        "THANK":  ["YOU"],
        "MORE":   ["WATER", "FOOD", "PLEASE"],
        "NO":     ["MORE", "PAIN", "STOP"],
        "CALL":   ["DOCTOR", "HELP", "POLICE"],
        "HELP":   ["ME", "PLEASE", "NOW"],
        "FEEL":   ["PAIN", "SICK", "GOOD", "BAD", "TIRED", "HUNGRY"],
    }

    def __init__(
        self,
        window: int = 7,
        min_votes: int = 4,
        stable_frames: int = 5,
        context_bonus: float = 0.08,
        confidence_threshold: float = 0.70,
    ):
        self._window = window
        self._min_votes = min_votes
        self._stable_frames = stable_frames
        self._context_bonus = context_bonus
        self._threshold = confidence_threshold

        self._buffer: deque = deque(maxlen=window)
        self._conf_buffer: deque = deque(maxlen=window)

        # Stability tracking
        self._stable_label: Optional[str] = None
        self._stable_count: int = 0

        # Context
        self._last_confirmed: Optional[str] = None
        self._confirmed_history: List[str] = []

        # Cooldown — prevent surfacing the same prediction twice quickly
        self._last_surfaced: str = ""
        self._last_surface_time: float = 0.0
        self._surface_cooldown: float = 1.5   # seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(self, label: str, confidence: float):
        """Feed one raw frame prediction."""
        self._buffer.append(label)
        self._conf_buffer.append(confidence)

    def get_stable_prediction(self) -> Tuple[Optional[str], float, str]:
        """
        Returns (label, effective_confidence, status_string).

        status_string ∈ {"READY", "STABILIZING", "LOW_CONF", "EMPTY"}
        """
        if len(self._buffer) == 0:
            return None, 0.0, "EMPTY"

        # --- Majority vote ---
        counts = Counter(self._buffer)
        majority_label, majority_count = counts.most_common(1)[0]

        # Average confidence for the majority label
        paired = zip(self._buffer, self._conf_buffer)
        majority_confs = [c for l, c in paired if l == majority_label]
        avg_conf = sum(majority_confs) / len(majority_confs) if majority_confs else 0.0

        # --- Context boost ---
        effective_conf = avg_conf + self._context_boost(majority_label)
        effective_conf = min(effective_conf, 1.0)

        # --- Threshold check ---
        if effective_conf < self._threshold:
            self._reset_stability()
            return majority_label, effective_conf, "LOW_CONF"

        # --- Stability gate ---
        if majority_count < self._min_votes:
            self._reset_stability()
            return majority_label, effective_conf, "STABILIZING"

        if majority_label == self._stable_label:
            self._stable_count += 1
        else:
            self._stable_label = majority_label
            self._stable_count = 1

        if self._stable_count < self._stable_frames:
            return majority_label, effective_conf, "STABILIZING"

        # --- Cooldown check ---
        now = time.time()
        if (majority_label == self._last_surfaced and
                now - self._last_surface_time < self._surface_cooldown):
            return majority_label, effective_conf, "COOLDOWN"

        self._last_surfaced = majority_label
        self._last_surface_time = now
        return majority_label, effective_conf, "READY"

    def confirm(self, label: str):
        """Call when user accepts a prediction (SPACE pressed)."""
        self._last_confirmed = label.upper()
        self._confirmed_history.append(self._last_confirmed)
        self._reset_stability()
        self._buffer.clear()
        self._conf_buffer.clear()

    def delete_last_confirmed(self):
        """Undo last confirmed word."""
        if self._confirmed_history:
            self._confirmed_history.pop()
            self._last_confirmed = (self._confirmed_history[-1]
                                    if self._confirmed_history else None)

    def reset(self):
        """Full reset."""
        self._buffer.clear()
        self._conf_buffer.clear()
        self._stable_label = None
        self._stable_count = 0
        self._last_confirmed = None
        self._confirmed_history.clear()
        self._last_surfaced = ""
        self._last_surface_time = 0.0

    @property
    def window_fill(self) -> float:
        """0.0–1.0 how full the window is."""
        return len(self._buffer) / self._window

    @property
    def last_confirmed(self) -> Optional[str]:
        return self._last_confirmed

    @property
    def context_suggestions(self) -> List[str]:
        """Next-word suggestions based on last confirmed word."""
        if self._last_confirmed:
            return self._BIGRAMS.get(self._last_confirmed, [])
        return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _context_boost(self, label: str) -> float:
        if not self._last_confirmed:
            return 0.0
        suggestions = self._BIGRAMS.get(self._last_confirmed, [])
        if label.upper() in suggestions:
            return self._context_bonus
        return 0.0

    def _reset_stability(self):
        self._stable_label = None
        self._stable_count = 0
