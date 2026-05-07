"""
logic.py — Enhanced fight detection and threat-level engine.

Intelligent, stable, and explainable movement-based behavior analysis.

Pipeline:
  1. Track previous landmarks for key joints only (wrists + ankles)
  2. Compute per-joint velocity via Euclidean displacement
  3. Build a weighted movement score (arms ×2, ankles ×1)
  4. Apply temporal smoothing (rolling average over 5 frames)
  5. Compute proximity from YOLO bounding boxes
  6. Detect sustained aggressive motion over multiple frames
  7. Combine into a final fight_score
  8. Return explainable reason string

CONSTRAINTS:
  - No deep learning models
  - Only 4 landmarks used: wrists (15, 16) and ankles (27, 28)
  - Lightweight computation for real-time performance
"""

import math
from collections import deque


# ──────────────────────────────────────────────────────────────────────────────
# Key joint indices (MediaPipe Pose)
# ──────────────────────────────────────────────────────────────────────────────
LM_LEFT_WRIST  = 15
LM_RIGHT_WRIST = 16
LM_LEFT_ANKLE  = 27
LM_RIGHT_ANKLE = 28

# ──────────────────────────────────────────────────────────────────────────────
# Configuration / thresholds
# ──────────────────────────────────────────────────────────────────────────────
FIGHT_SCORE_THRESHOLD   = 0.55   # fight_score above this → fight_detected

# Weights for the final fight_score combination
MOVEMENT_WEIGHT         = 0.35   # smoothed movement score
PROXIMITY_WEIGHT        = 0.35   # how close people are
MOTION_WEIGHT           = 0.15   # pixel-level motion from motion.py
AGGRESSION_WEIGHT       = 0.15   # sustained aggressive motion bonus

# Speed normalisers — raw pixel displacement above these ≈ 1.0 (aggressive)
ARM_SPEED_NORM          = 35.0   # pixels/frame for wrists
LEG_SPEED_NORM          = 25.0   # pixels/frame for ankles

# Movement score weights (arms are weighted 2× more than legs)
WRIST_VELOCITY_WEIGHT   = 2.0
ANKLE_VELOCITY_WEIGHT   = 1.0

# Proximity: centre-to-centre distance below this = "close"
PROXIMITY_CLOSE_PX      = 150.0

# Temporal smoothing buffer size
SMOOTHING_WINDOW        = 5

# Aggressive motion: if avg movement stays above this for N frames
AGGRESSION_THRESHOLD    = 0.40
AGGRESSION_FRAME_COUNT  = 3      # consecutive frames above threshold

# Minimum signal gates for fight detection (prevents isolated spikes)
MIN_MOVEMENT_FOR_FIGHT  = 0.15
MIN_PROXIMITY_FOR_FIGHT = 0.20


# ──────────────────────────────────────────────────────────────────────────────
# Pure helper functions
# ──────────────────────────────────────────────────────────────────────────────

def compute_velocity(prev_pos: tuple | None, curr_pos: tuple | None) -> float:
    """
    Compute Euclidean displacement between two (x, y) positions.

    Returns 0.0 if either position is None (first-frame safety).
    """
    if prev_pos is None or curr_pos is None:
        return 0.0
    return math.sqrt((curr_pos[0] - prev_pos[0]) ** 2
                     + (curr_pos[1] - prev_pos[1]) ** 2)


def compute_movement_score(wrist_velocity: float, ankle_velocity: float) -> float:
    """
    Weighted combination of wrist and ankle velocities.

    Arms are weighted 2× because punching/striking motions are
    the strongest indicator of aggressive behavior.

    Returns a normalised score in [0, 1].
    """
    # Normalise each velocity to [0, 1] range
    norm_wrist = min(wrist_velocity / ARM_SPEED_NORM, 1.0)
    norm_ankle = min(ankle_velocity / LEG_SPEED_NORM, 1.0)

    # Weighted sum, then normalise by total weight
    total_weight = WRIST_VELOCITY_WEIGHT + ANKLE_VELOCITY_WEIGHT
    raw = (WRIST_VELOCITY_WEIGHT * norm_wrist
           + ANKLE_VELOCITY_WEIGHT * norm_ankle)
    return min(raw / total_weight, 1.0)


def detect_aggressive_motion(history: deque) -> tuple[bool, float]:
    """
    Check if high movement has been sustained over multiple consecutive frames.

    Scans the tail of the history buffer. If the last AGGRESSION_FRAME_COUNT
    values all exceed AGGRESSION_THRESHOLD, aggressive motion is detected.

    Returns:
        (is_aggressive, aggression_score)
        aggression_score is 0.0 or the average of the aggressive tail.
    """
    if len(history) < AGGRESSION_FRAME_COUNT:
        return False, 0.0

    # Check the most recent N frames
    tail = list(history)[-AGGRESSION_FRAME_COUNT:]
    if all(v > AGGRESSION_THRESHOLD for v in tail):
        return True, sum(tail) / len(tail)
    return False, 0.0


# ──────────────────────────────────────────────────────────────────────────────
# FightDetector class
# ──────────────────────────────────────────────────────────────────────────────

class FightDetector:
    """
    Enhanced rule-based fight / aggressive-behaviour detector.

    Uses landmark velocity tracking, temporal smoothing, proximity analysis,
    and sustained-aggression detection to produce a stable, explainable
    fight score.
    """

    def __init__(self):
        # Previous frame landmark positions: (x, y) or None
        self._prev_landmarks: dict[int, tuple | None] = {
            LM_LEFT_WRIST: None,
            LM_RIGHT_WRIST: None,
            LM_LEFT_ANKLE: None,
            LM_RIGHT_ANKLE: None,
        }

        # Rolling buffer for temporal smoothing of movement scores
        self._movement_history: deque = deque(maxlen=SMOOTHING_WINDOW)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _get_lm(landmarks: list[dict] | None, lm_id: int) -> tuple | None:
        """Extract (x, y) for a landmark by ID. Returns None if missing."""
        if landmarks is None:
            return None
        for lm in landmarks:
            if lm["id"] == lm_id:
                return (lm["x"], lm["y"])
        return None

    def _compute_joint_velocities(self, landmarks: list[dict] | None) -> dict:
        """
        Compute per-joint velocity for the 4 key joints.

        Compares current positions against stored previous positions,
        then updates the stored positions for the next frame.

        Returns dict with raw velocities (pixels/frame) for each joint.
        """
        velocities = {}
        for lm_id in [LM_LEFT_WRIST, LM_RIGHT_WRIST, LM_LEFT_ANKLE, LM_RIGHT_ANKLE]:
            curr_pos = self._get_lm(landmarks, lm_id)
            prev_pos = self._prev_landmarks[lm_id]

            # Compute velocity (handles None safely)
            velocities[lm_id] = compute_velocity(prev_pos, curr_pos)

            # Update stored position for next frame
            self._prev_landmarks[lm_id] = curr_pos

        return velocities

    @staticmethod
    def _compute_proximity(detections: list[dict]) -> float:
        """
        Compute proximity score from YOLO bounding boxes.

        Finds the minimum centre-to-centre distance between detected persons.
        Returns a normalised value in [0, 1] — 1.0 means very close.
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

        # Invert: closer → higher score, clamped to [0, 1]
        proximity_score = max(0.0, 1.0 - (min_dist / PROXIMITY_CLOSE_PX))
        return min(proximity_score, 1.0)

    @staticmethod
    def _count_people(detections: list[dict]) -> int:
        return sum(1 for d in detections if d["label"] == "Person")

    def _build_reason(self, arm_speed: float, leg_speed: float,
                      proximity: float, is_aggressive: bool,
                      people_close: bool) -> str:
        """
        Build an explainable reason string describing WHY a fight was detected.

        Combines the dominant signals into a human-readable explanation.
        """
        reasons = []

        # Arm movement explanation
        if arm_speed > 0.5:
            reasons.append("High arm velocity")
        elif arm_speed > 0.25:
            reasons.append("Moderate arm movement")

        # Leg movement explanation
        if leg_speed > 0.5:
            reasons.append("rapid leg movement")
        elif leg_speed > 0.25:
            reasons.append("notable leg movement")

        # Proximity explanation
        if people_close:
            reasons.append("close proximity between persons")

        # Sustained aggression explanation
        if is_aggressive:
            reasons.append("sustained aggressive motion")

        if not reasons:
            return "Aggressive interaction between people detected"

        # Capitalise first word, join with " + "
        result = " + ".join(reasons)
        return result[0].upper() + result[1:]

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
        landmarks : list[dict] | None
            Pose landmarks from MediaPipe (first person).
        detections : list[dict]
            YOLO detections with bounding boxes.
        motion_score : float
            Pixel-level motion score from motion.py (0–1).
        threshold_override : float | None
            If provided, overrides FIGHT_SCORE_THRESHOLD.
            Used by the feedback-driven auto-calibrator.

        Returns
        -------
        dict with keys:
            arm_speed, leg_speed, proximity, motion_intensity,
            fight_score, fight_detected, people_close, reason,
            movement_score, aggressive_motion
        """

        # ── Step 1–2: Compute per-joint velocities (only 4 key joints) ──
        velocities = self._compute_joint_velocities(landmarks)

        # Average wrist velocity (raw pixels/frame)
        raw_wrist_vel = (velocities[LM_LEFT_WRIST] + velocities[LM_RIGHT_WRIST]) / 2.0
        # Average ankle velocity (raw pixels/frame)
        raw_ankle_vel = (velocities[LM_LEFT_ANKLE] + velocities[LM_RIGHT_ANKLE]) / 2.0

        # ── Step 3: Compute weighted movement score ──
        movement_score = compute_movement_score(raw_wrist_vel, raw_ankle_vel)

        # ── Step 4: Temporal smoothing (rolling average of last 5 frames) ──
        self._movement_history.append(movement_score)
        if len(self._movement_history) > 0:
            smoothed_movement = sum(self._movement_history) / len(self._movement_history)
        else:
            smoothed_movement = movement_score

        # ── Step 5: Proximity from YOLO bounding boxes ──
        proximity = self._compute_proximity(detections)
        people = self._count_people(detections)
        people_close = people >= 2 and proximity > MIN_PROXIMITY_FOR_FIGHT

        # ── Step 6: Sustained aggressive motion detection ──
        is_aggressive, aggression_score = detect_aggressive_motion(self._movement_history)

        # ── Step 7: Normalised arm/leg speeds for display (0–1) ──
        arm_speed = min(raw_wrist_vel / ARM_SPEED_NORM, 1.0)
        leg_speed = min(raw_ankle_vel / LEG_SPEED_NORM, 1.0)

        # ── Step 8: Final fight score ──
        #   Combines smoothed movement, proximity, motion, and aggression bonus
        fight_score = (
            MOVEMENT_WEIGHT   * smoothed_movement
            + PROXIMITY_WEIGHT  * proximity
            + MOTION_WEIGHT     * motion_score
            + AGGRESSION_WEIGHT * aggression_score
        )
        fight_score = min(fight_score, 1.0)

        # ── Step 9: Fight detection condition ──
        #   Requires: score above threshold + 2+ people close + meaningful movement
        effective_threshold = (threshold_override
                               if threshold_override is not None
                               else FIGHT_SCORE_THRESHOLD)
        fight_detected = (
            fight_score > effective_threshold
            and people >= 2
            and people_close
            and smoothed_movement > MIN_MOVEMENT_FOR_FIGHT
        )

        # ── Step 10: Explainability — build reason string ──
        reason = ""
        if fight_detected:
            reason = self._build_reason(
                arm_speed, leg_speed, proximity, is_aggressive, people_close
            )

        return {
            "arm_speed":          round(arm_speed, 3),
            "leg_speed":          round(leg_speed, 3),
            "proximity":          round(proximity, 3),
            "motion_intensity":   round(motion_score, 3),
            "fight_score":        round(fight_score, 3),
            "fight_detected":     fight_detected,
            "people_close":       people_close,
            "movement_score":     round(smoothed_movement, 3),
            "aggressive_motion":  is_aggressive,
            "reason":             reason,
        }

    def reset(self):
        """Reset all internal state (e.g. when switching video sources)."""
        for k in self._prev_landmarks:
            self._prev_landmarks[k] = None
        self._movement_history.clear()


# ──────────────────────────────────────────────────────────────────────────────
# Threat-level engine
# ──────────────────────────────────────────────────────────────────────────────

def compute_threat_level(weapon_detected: bool, fight_detected: bool,
                         people_count: int = 0,
                         fight_reason: str = "") -> tuple[str, str]:
    """
    Determine the overall threat level.

    Rules
    -----
    • weapon + fight           → CRITICAL
    • weapon only              → HIGH
    • fight (2+ people, close, sustained movement) → MEDIUM
    • none                     → SAFE
    """
    if weapon_detected and fight_detected:
        reason = fight_reason if fight_reason else "Weapon + violent motion detected"
        return "CRITICAL", reason
    elif weapon_detected:
        return "HIGH", "Weapon detected in frame"
    elif fight_detected:
        reason = fight_reason if fight_reason else "Aggressive interaction between people detected"
        return "MEDIUM", reason
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
