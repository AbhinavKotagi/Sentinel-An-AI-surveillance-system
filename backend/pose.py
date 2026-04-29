"""
pose.py — Human pose estimation using MediaPipe Tasks API (0.10.30+).

Uses PoseLandmarker with VIDEO running mode for real-time skeleton detection.
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

# Path to the pose landmarker model
POSE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "pose_landmarker_lite.task")

# MediaPipe Pose connections for drawing skeleton
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),
    (12,14),(14,16),(16,18),(16,20),(16,22),(11,23),(12,24),
    (23,24),(23,25),(25,27),(27,29),(27,31),(24,26),(26,28),
    (28,30),(28,32),
]


class PoseEstimator:
    """Wraps MediaPipe PoseLandmarker for real-time skeleton detection."""

    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    NOSE = 0

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.landmarker = None
        self._frame_ts = 0

        if not _MP_AVAILABLE:
            print("[ERROR] Cannot init PoseEstimator — mediapipe not installed.")
            return

        if not os.path.isfile(POSE_MODEL_PATH):
            print(f"[WARNING] Pose model not found: {POSE_MODEL_PATH}")
            print("          Download pose_landmarker_lite.task into backend/models/")
            return

        try:
            base_options = mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
            options = mp_vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.VIDEO,
                num_poses=3,
                min_pose_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)
            print("[INFO] MediaPipe PoseLandmarker initialised (lite model)")
        except Exception as exc:
            print(f"[ERROR] Failed to init PoseLandmarker: {exc}")

    def process(self, frame: np.ndarray) -> dict:
        """
        Process a BGR frame. Returns dict with:
            landmarks: list[dict] | None  — pixel coords {id, x, y, z, visibility}
            frame: np.ndarray             — frame with skeleton overlay
        """
        out_frame = frame.copy()
        h, w = frame.shape[:2]

        if self.landmarker is None:
            return {"landmarks": None, "frame": out_frame, "raw_landmarks": None}

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._frame_ts += 33  # ~30fps timestamp in ms
        try:
            result = self.landmarker.detect_for_video(mp_image, self._frame_ts)
        except Exception:
            return {"landmarks": None, "frame": out_frame, "raw_landmarks": None}

        landmarks_list = None
        raw = None

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            raw = result.pose_landmarks[0]  # first person
            landmarks_list = []
            for idx, lm in enumerate(raw):
                landmarks_list.append({
                    "id": idx,
                    "x": int(lm.x * w),
                    "y": int(lm.y * h),
                    "z": lm.z,
                    "visibility": lm.visibility,
                })

            # Draw all detected pose skeletons
            for pose_lms in result.pose_landmarks:
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in pose_lms]
                # Draw connections
                for c1, c2 in POSE_CONNECTIONS:
                    if c1 < len(pts) and c2 < len(pts):
                        cv2.line(out_frame, pts[c1], pts[c2], (0, 255, 163), 2)
                # Draw landmarks
                for px, py in pts:
                    cv2.circle(out_frame, (px, py), 4, (0, 200, 255), -1)

        return {
            "landmarks": landmarks_list,
            "frame": out_frame,
            "raw_landmarks": raw,
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
