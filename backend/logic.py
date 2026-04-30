"""
logic.py — Fight detection and threat-level engine.

Computes:
  • arm_speed      – pixel velocity of wrist landmarks between frames
  • leg_speed      – pixel velocity of ankle landmarks between frames
  • proximity      – minimum distance between person bounding boxes
  • motion_score   – from motion.py
  • fight_score    – weighted combination of above signals
  • threat_level   – SAFE / MEDIUM / HIGH / CRITICAL

MEDIUM threat requires:
  - No weapon
  - 2+ people in close proximity
  - Distinct arm AND leg movements
"""

import math
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Configuration / thresholds
# ──────────────────────────────────────────────────────────────────────────────
FIGHT_SCORE_THRESHOLD = 0.55        # fight_score above this → fight_detected
ARM_SPEED_WEIGHT      = 0.30
LEG_SPEED_WEIGHT      = 0.20
PROXIMITY_WEIGHT      = 0.30
MOTION_WEIGHT         = 0.20

# Speed above this (pixels/frame) is considered aggressive
ARM_SPEED_AGGRO       = 35.0
LEG_SPEED_AGGRO       = 25.0

# Proximity below this (pixels) between two persons is "close"
PROXIMITY_CLOSE       = 150.0

# Minimum thresholds for MEDIUM — both must be exceeded
MIN_ARM_SPEED_FOR_FIGHT = 0.15     # normalised 0–1
MIN_LEG_SPEED_FOR_FIGHT = 0.10     # normalised 0–1
MIN_PROXIMITY_FOR_FIGHT = 0.20     # normalised 0–1 (means they're close)


class FightDetector:
    """Rule-based fight / aggressive-behaviour detector."""

    def __init__(self):
        self._prev_left_wrist  = None
        self._prev_right_wrist = None
        self._prev_left_ankle  = None
        self._prev_right_ankle = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _dist(p1: tuple, p2: tuple) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    @staticmethod
    def _get_lm(landmarks, lm_id):
        """Get a landmark by ID from the list."""
        if landmarks is None:
            return None
        return next((lm for lm in landmarks if lm["id"] == lm_id), None)

    def _compute_limb_speed(self, landmarks, lm_id_left, lm_id_right,
                            prev_left, prev_right, aggro_threshold):
        """Compute average speed of a limb pair (wrists or ankles)."""
        left = self._get_lm(landmarks, lm_id_left)
        right = self._get_lm(landmarks, lm_id_right)

        speed = 0.0
        count = 0

        if left and prev_left:
            speed += self._dist((left["x"], left["y"]), prev_left)
            count += 1
        if right and prev_right:
            speed += self._dist((right["x"], right["y"]), prev_right)
            count += 1

        new_prev_left = (left["x"], left["y"]) if left else None
        new_prev_right = (right["x"], right["y"]) if right else None

        if count == 0:
            return 0.0, new_prev_left, new_prev_right

        avg_speed = speed / count
        return min(avg_speed / aggro_threshold, 1.0), new_prev_left, new_prev_right

    def _compute_arm_speed(self, landmarks):
        # Wrist IDs: left=15, right=16
        speed, self._prev_left_wrist, self._prev_right_wrist = \
            self._compute_limb_speed(landmarks, 15, 16,
                                     self._prev_left_wrist, self._prev_right_wrist,
                                     ARM_SPEED_AGGRO)
        return speed

    def _compute_leg_speed(self, landmarks):
        # Ankle IDs: left=27, right=28
        speed, self._prev_left_ankle, self._prev_right_ankle = \
            self._compute_limb_speed(landmarks, 27, 28,
                                     self._prev_left_ankle, self._prev_right_ankle,
                                     LEG_SPEED_AGGRO)
        return speed

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

        proximity_score = max(0.0, 1.0 - (min_dist / PROXIMITY_CLOSE))
        return min(proximity_score, 1.0)

    @staticmethod
    def _count_people(detections: list[dict]) -> int:
        return sum(1 for d in detections if d["label"] == "Person")

    # ------------------------------------------------------------------
    # Public API — fight detection
    # ------------------------------------------------------------------
    def detect_fight(
        self,
        landmarks: list[dict] | None,
        detections: list[dict],
        motion_score: float,
        threshold_override: float | None = None,
    ) -> dict:
        """
        Compute fight signals and determine if a fight is occurring.

        Parameters
        ----------
        threshold_override : float | None
            If provided, overrides FIGHT_SCORE_THRESHOLD for this call.
            Used by the feedback-driven auto-calibrator.

        Fight requires ALL of:
          1. proximity > MIN_PROXIMITY_FOR_FIGHT  (2+ people close)
          2. arm_speed > MIN_ARM_SPEED_FOR_FIGHT  (distinct arm movement)
          3. leg_speed > MIN_LEG_SPEED_FOR_FIGHT  (distinct leg movement)
          4. fight_score > FIGHT_SCORE_THRESHOLD
        """
        arm_speed  = self._compute_arm_speed(landmarks)
        leg_speed  = self._compute_leg_speed(landmarks)
        proximity  = self._compute_proximity(detections)
        people     = self._count_people(detections)
        motion_int = motion_score

        fight_score = (
            ARM_SPEED_WEIGHT * arm_speed
            + LEG_SPEED_WEIGHT * leg_speed
            + PROXIMITY_WEIGHT * proximity
            + MOTION_WEIGHT    * motion_int
        )

        # Strict fight detection: need close proximity + distinct limb movement
        effective_threshold = threshold_override if threshold_override is not None else FIGHT_SCORE_THRESHOLD
        fight_detected = (
            fight_score > effective_threshold
            and people >= 2
            and proximity > MIN_PROXIMITY_FOR_FIGHT
            and arm_speed > MIN_ARM_SPEED_FOR_FIGHT
            and leg_speed > MIN_LEG_SPEED_FOR_FIGHT
        )

        return {
            "arm_speed":        round(arm_speed, 3),
            "leg_speed":        round(leg_speed, 3),
            "proximity":        round(proximity, 3),
            "motion_intensity": round(motion_int, 3),
            "fight_score":      round(fight_score, 3),
            "fight_detected":   fight_detected,
            "people_close":     people >= 2 and proximity > MIN_PROXIMITY_FOR_FIGHT,
        }

    def reset(self):
        """Reset internal state."""
        self._prev_left_wrist  = None
        self._prev_right_wrist = None
        self._prev_left_ankle  = None
        self._prev_right_ankle = None


# ──────────────────────────────────────────────────────────────────────────────
# Threat-level engine
# ──────────────────────────────────────────────────────────────────────────────

def compute_threat_level(weapon_detected: bool, fight_detected: bool,
                         people_count: int = 0) -> tuple[str, str]:
    """
    Determine the overall threat level.

    Rules
    -----
    • weapon + fight           → CRITICAL
    • weapon only              → HIGH
    • fight (2+ people, close, limb movement) → MEDIUM
    • none                     → SAFE
    """
    if weapon_detected and fight_detected:
        return "CRITICAL", "Weapon + violent motion detected"
    elif weapon_detected:
        return "HIGH", "Weapon detected in frame"
    elif fight_detected:
        return "MEDIUM", "Aggressive interaction between people detected"
    else:
        return "SAFE", ""


def classify_detection_type(weapon_detected: bool, fight_detected: bool) -> str:
    """Return the detection category for feedback tracking."""
    if weapon_detected and fight_detected:
        return "combined"
    elif weapon_detected:
        return "weapon"
    elif fight_detected:
        return "fight"
    return "none"
