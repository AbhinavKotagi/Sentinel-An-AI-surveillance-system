"""
logic.py — Fight detection and threat-level engine.

Computes:
  • arm_speed      – pixel velocity of wrist landmarks between frames
  • proximity      – minimum distance between person bounding boxes
  • motion_score   – from motion.py
  • fight_score    – weighted combination of above signals
  • threat_level   – SAFE / MEDIUM / HIGH / CRITICAL
"""

import math
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Configuration / thresholds
# ──────────────────────────────────────────────────────────────────────────────
FIGHT_SCORE_THRESHOLD = 0.55        # fight_score above this → fight_detected
ARM_SPEED_WEIGHT      = 0.40
PROXIMITY_WEIGHT      = 0.30
MOTION_WEIGHT         = 0.30

# Arm speed above this (pixels/frame) is considered aggressive
ARM_SPEED_AGGRO       = 35.0

# Proximity below this (pixels) between two persons is "close"
PROXIMITY_CLOSE       = 150.0


class FightDetector:
    """Rule-based fight / aggressive-behaviour detector."""

    def __init__(self):
        # Store previous wrist positions for velocity calculation
        self._prev_left_wrist  = None   # (x, y)
        self._prev_right_wrist = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _dist(p1: tuple, p2: tuple) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def _compute_arm_speed(self, landmarks: list[dict] | None) -> float:
        """
        Average pixel-distance moved by both wrists since last frame.
        Returns a normalised value in [0, 1].
        """
        if landmarks is None:
            self._prev_left_wrist = None
            self._prev_right_wrist = None
            return 0.0

        # MediaPipe landmark IDs
        LW, RW = 15, 16
        left  = next((lm for lm in landmarks if lm["id"] == LW), None)
        right = next((lm for lm in landmarks if lm["id"] == RW), None)

        speed = 0.0
        count = 0

        if left and self._prev_left_wrist:
            speed += self._dist((left["x"], left["y"]), self._prev_left_wrist)
            count += 1
        if right and self._prev_right_wrist:
            speed += self._dist((right["x"], right["y"]), self._prev_right_wrist)
            count += 1

        # Update state
        self._prev_left_wrist  = (left["x"],  left["y"])  if left  else None
        self._prev_right_wrist = (right["x"], right["y"]) if right else None

        if count == 0:
            return 0.0

        avg_speed = speed / count
        return min(avg_speed / ARM_SPEED_AGGRO, 1.0)

    @staticmethod
    def _compute_proximity(detections: list[dict]) -> float:
        """
        Minimum centre-to-centre distance between person bounding boxes.
        Returns normalised value in [0, 1] — 1 means very close.
        """
        persons = [d for d in detections if d["label"] == "Person"]
        if len(persons) < 2:
            return 0.0

        min_dist = float("inf")
        for i in range(len(persons)):
            bx1 = persons[i]["box"]
            cx1 = (bx1[0] + bx1[2]) / 2
            cy1 = (bx1[1] + bx1[3]) / 2
            for j in range(i + 1, len(persons)):
                bx2 = persons[j]["box"]
                cx2 = (bx2[0] + bx2[2]) / 2
                cy2 = (bx2[1] + bx2[3]) / 2
                d = math.hypot(cx1 - cx2, cy1 - cy2)
                if d < min_dist:
                    min_dist = d

        # Invert: closer → higher score
        proximity_score = max(0.0, 1.0 - (min_dist / PROXIMITY_CLOSE))
        return min(proximity_score, 1.0)

    # ------------------------------------------------------------------
    # Public API — fight detection
    # ------------------------------------------------------------------
    def detect_fight(
        self,
        landmarks: list[dict] | None,
        detections: list[dict],
        motion_score: float,
    ) -> dict:
        """
        Compute fight signals and return result dict.

        Returns
        -------
        dict:
            arm_speed        : float  (0–1)
            proximity        : float  (0–1)
            motion_intensity : float  (0–1)
            fight_score      : float  (0–1)
            fight_detected   : bool
        """
        arm_speed  = self._compute_arm_speed(landmarks)
        proximity  = self._compute_proximity(detections)
        motion_int = motion_score  # already 0–1

        fight_score = (
            ARM_SPEED_WEIGHT * arm_speed
            + PROXIMITY_WEIGHT * proximity
            + MOTION_WEIGHT   * motion_int
        )

        return {
            "arm_speed":        round(arm_speed, 3),
            "proximity":        round(proximity, 3),
            "motion_intensity": round(motion_int, 3),
            "fight_score":      round(fight_score, 3),
            "fight_detected":   fight_score > FIGHT_SCORE_THRESHOLD,
        }

    def reset(self):
        """Reset internal state."""
        self._prev_left_wrist  = None
        self._prev_right_wrist = None


# ──────────────────────────────────────────────────────────────────────────────
# Threat-level engine
# ──────────────────────────────────────────────────────────────────────────────

def compute_threat_level(weapon_detected: bool, fight_detected: bool) -> tuple[str, str]:
    """
    Determine the overall threat level.

    Rules
    -----
    • weapon + fight   → CRITICAL
    • weapon only      → HIGH
    • fight only       → MEDIUM
    • none             → SAFE

    Returns
    -------
    (threat_level, reason)
    """
    if weapon_detected and fight_detected:
        return "CRITICAL", "Weapon + violent motion detected"
    elif weapon_detected:
        return "HIGH", "Weapon detected in frame"
    elif fight_detected:
        return "MEDIUM", "Aggressive / fight-like behaviour detected"
    else:
        return "SAFE", ""
