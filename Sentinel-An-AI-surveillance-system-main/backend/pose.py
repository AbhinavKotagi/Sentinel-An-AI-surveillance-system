"""
pose.py — Multi-person pose estimation using MediaPipe Tasks API (0.10.30+).

Uses PoseLandmarker with VIDEO running mode.
Returns landmarks for ALL detected persons, not just the first.
"""

import os
import cv2
import numpy as np
import time

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False
    print("[WARNING] mediapipe not installed. Run: pip install mediapipe")

POSE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "pose_landmarker_lite.task")

# MediaPipe Pose skeleton connections (33 landmarks)
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),
    (12,14),(14,16),(16,18),(16,20),(16,22),(11,23),(12,24),
    (23,24),(23,25),(25,27),(27,29),(27,31),(24,26),(26,28),
    (28,30),(28,32),
]

# Different colours for each person's skeleton
SKELETON_COLOURS = [
    (0, 255, 163),   # green-cyan
    (255, 165, 0),   # orange
    (255, 0, 200),   # pink
    (100, 200, 255), # light blue
    (200, 255, 0),   # yellow-green
]

JOINT_COLOUR = (0, 200, 255)  # yellow for all joints


class PoseEstimator:
    """Multi-person MediaPipe PoseLandmarker."""

    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    NOSE = 0

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.landmarker = None
        self._start_time = time.time()

        if not _MP_AVAILABLE:
            print("[ERROR] Cannot init PoseEstimator — mediapipe not installed.")
            return

        if not os.path.isfile(POSE_MODEL_PATH):
            print(f"[WARNING] Pose model not found: {POSE_MODEL_PATH}")
            return

        try:
            base_options = mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
            options = mp_vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.VIDEO,
                num_poses=5,
                min_pose_detection_confidence=min_detection_confidence,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=min_tracking_confidence,
            )
            self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)
            print("[INFO] MediaPipe PoseLandmarker initialised (num_poses=5)")
        except Exception as exc:
            print(f"[ERROR] Failed to init PoseLandmarker: {exc}")

    def process(self, frame: np.ndarray) -> dict:
        """
        Process a BGR frame.

        Returns dict with:
            landmarks      : list[dict] | None — first person's landmarks for fight logic
            all_landmarks  : list[list[dict]]  — ALL persons' landmarks
            frame          : np.ndarray        — frame with skeleton overlays
            num_poses      : int               — number of poses detected
        """
        out_frame = frame.copy()
        h, w = frame.shape[:2]

        if self.landmarker is None:
            return {
                "landmarks": None, "all_landmarks": [],
                "frame": out_frame, "raw_landmarks": None, "num_poses": 0,
            }

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Use real elapsed time in milliseconds (avoids timestamp drift/glitching)
        timestamp_ms = int((time.time() - self._start_time) * 1000)

        try:
            result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception:
            return {
                "landmarks": None, "all_landmarks": [],
                "frame": out_frame, "raw_landmarks": None, "num_poses": 0,
            }

        all_landmarks = []
        first_landmarks = None
        num_poses = 0

        if result.pose_landmarks:
            num_poses = len(result.pose_landmarks)

            for person_idx, pose_lms in enumerate(result.pose_landmarks):
                # Convert to pixel coordinates
                person_landmarks = []
                pts = []
                for idx, lm in enumerate(pose_lms):
                    px, py = int(lm.x * w), int(lm.y * h)
                    person_landmarks.append({
                        "id": idx,
                        "x": px, "y": py,
                        "z": lm.z,
                        "visibility": lm.visibility,
                    })
                    pts.append((px, py))

                all_landmarks.append(person_landmarks)
                if first_landmarks is None:
                    first_landmarks = person_landmarks

                # Draw skeleton with per-person colour
                # [OPTIMIZATION] Thinner lines for faster rendering
                skel_color = SKELETON_COLOURS[person_idx % len(SKELETON_COLOURS)]
                for c1, c2 in POSE_CONNECTIONS:
                    if c1 < len(pts) and c2 < len(pts):
                        # Only draw if both landmarks are visible enough
                        v1 = pose_lms[c1].visibility
                        v2 = pose_lms[c2].visibility
                        if v1 > 0.3 and v2 > 0.3:
                            cv2.line(out_frame, pts[c1], pts[c2], skel_color, 1, cv2.LINE_AA)

                # Draw joints — smaller circles
                for idx, (px, py) in enumerate(pts):
                    if pose_lms[idx].visibility > 0.3:
                        cv2.circle(out_frame, (px, py), 3, JOINT_COLOUR, -1)

        return {
            "landmarks": first_landmarks,
            "all_landmarks": all_landmarks,
            "frame": out_frame,
            "raw_landmarks": result.pose_landmarks[0] if result.pose_landmarks else None,
            "num_poses": num_poses,
        }

    @staticmethod
    def get_joint(landmarks, joint_id):
        if landmarks is None:
            return None
        for lm in landmarks:
            if lm["id"] == joint_id:
                return lm
        return None

    def release(self):
        if self.landmarker is not None:
            self.landmarker.close()
