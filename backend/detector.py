"""
detector.py — YOLO-based object detection (people, knife, gun).

Uses ultralytics YOLOv8 with a custom-trained weight file (models/best.pt).
Falls back gracefully if the model file is not found.
"""

import os
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Try to import ultralytics; provide clear error if missing
# ---------------------------------------------------------------------------
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    print("[WARNING] ultralytics not installed. Run: pip install ultralytics")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best.pt")

# Labels we care about (map from YOLO class names → our canonical labels)
TARGET_LABELS = {
    "person": "Person",
    "knife":  "Knife",
    "gun":    "Gun",
    # Common alternate names in custom datasets
    "pistol": "Gun",
    "weapon": "Gun",
    "rifle":  "Gun",
}

# Colour map for drawing (BGR)
LABEL_COLOURS = {
    "Person": (0, 255, 163),   # green-cyan
    "Knife":  (0, 100, 255),   # orange
    "Gun":    (0, 0, 255),     # red
}

DEFAULT_COLOUR = (255, 255, 255)


class ThreatDetector:
    """Wraps a YOLOv8 model for real-time object detection."""

    def __init__(self, model_path: str = MODEL_PATH, conf_threshold: float = 0.40):
        self.conf_threshold = conf_threshold
        self.model = None

        if YOLO is None:
            print("[ERROR] Cannot load YOLO — ultralytics is not installed.")
            return

        if not os.path.isfile(model_path):
            print(f"[WARNING] Model file not found: {model_path}")
            print("          Place your trained best.pt inside backend/models/ and restart.")
            print("          Detection will be DISABLED until the model is available.")
            return

        try:
            self.model = YOLO(model_path)
            print(f"[INFO] YOLO model loaded from {model_path}")
        except Exception as exc:
            print(f"[ERROR] Failed to load YOLO model: {exc}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> dict:
        """
        Run detection on a single BGR frame.

        Returns
        -------
        dict with keys:
            detections   : list[dict]  — each dict has label, confidence, box (x1,y1,x2,y2), color
            people_count : int
            weapon_detected : bool
        """
        detections: list[dict] = []
        people_count = 0
        weapon_detected = False

        if self.model is None:
            return {
                "detections": detections,
                "people_count": people_count,
                "weapon_detected": weapon_detected,
            }

        # Run inference (stream=False → returns a list with one Results object)
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            verbose=False,
        )

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                # Extract raw class name from the model
                cls_id = int(box.cls[0])
                raw_label = result.names.get(cls_id, "unknown").lower()

                # Map to our canonical label (skip if not a target)
                canonical = TARGET_LABELS.get(raw_label)
                if canonical is None:
                    continue

                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                color = LABEL_COLOURS.get(canonical, DEFAULT_COLOUR)

                detections.append({
                    "label":      canonical,
                    "confidence": round(conf, 2),
                    "box":        (x1, y1, x2, y2),
                    "color":      color,
                })

                if canonical == "Person":
                    people_count += 1
                elif canonical in ("Knife", "Gun"):
                    weapon_detected = True

        return {
            "detections":      detections,
            "people_count":    people_count,
            "weapon_detected": weapon_detected,
        }

    # ------------------------------------------------------------------
    # Drawing helper
    # ------------------------------------------------------------------
    @staticmethod
    def draw_detections(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        """Draw bounding boxes and labels onto a copy of the frame."""
        out = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            color = det["color"]
            label_text = f"{det['label']}  {det['confidence']:.0%}"

            # Main bounding box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # Corner accents
            cs = 14
            for cx, cy, dx, dy in [
                (x1, y1,  1,  1),
                (x2, y1, -1,  1),
                (x1, y2,  1, -1),
                (x2, y2, -1, -1),
            ]:
                cv2.line(out, (cx, cy), (cx + dx * cs, cy), color, 3)
                cv2.line(out, (cx, cy), (cx, cy + dy * cs), color, 3)

            # Label pill background
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, y1 - th - 12), (x1 + tw + 12, y1), color, -1)
            cv2.putText(
                out, label_text, (x1 + 6, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 1, cv2.LINE_AA,
            )

        return out
