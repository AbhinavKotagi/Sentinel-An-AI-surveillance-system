"""
detector.py — YOLO-based object detection (people, knife, gun).

Uses TWO models:
  • yolov8n.pt (COCO pretrained) — reliable person detection
  • best.pt (custom trained)     — weapon detection (knife, gun, etc.)

Falls back gracefully if either model is missing.
"""

import os
import cv2
import numpy as np
import torch  # Added for GPU support

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    print("[WARNING] ultralytics not installed. Run: pip install ultralytics")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
CUSTOM_MODEL_PATH = os.path.join(MODELS_DIR, "best.pt")
PERSON_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8n.pt")

# Weapon labels we look for in the custom model
WEAPON_LABELS = {"knife", "gun", "pistol", "weapon", "rifle", "sword", "bat"}

# Colour map for drawing (BGR)
LABEL_COLOURS = {
    "Person": (0, 255, 163),
    "Knife":  (0, 100, 255),
    "Gun":    (0, 0, 255),
}
DEFAULT_COLOUR = (255, 255, 255)


class ThreatDetector:
    """Uses YOLOv8n for people + custom model for weapons."""

    def __init__(self, model_path: str = CUSTOM_MODEL_PATH, conf_threshold: float = 0.40):
        self.conf_threshold = conf_threshold
        self.person_model = None
        self.weapon_model = None

        if YOLO is None:
            print("[ERROR] Cannot load YOLO — ultralytics is not installed.")
            return

        # ── Load person detection model (yolov8n — auto-downloads) ──
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.person_model = YOLO(PERSON_MODEL_PATH, device=device)
            print(f"[INFO] Person model loaded ({device}): {PERSON_MODEL_PATH}")
        except Exception:
            try:
                print("[INFO] Downloading yolov8n.pt for person detection...")
                self.person_model = YOLO("yolov8n.pt")
                print("[INFO] Person model (yolov8n) loaded successfully")
            except Exception as exc:
                print(f"[ERROR] Failed to load person model: {exc}")

        # ── Load custom weapon model ──
        if os.path.isfile(model_path):
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.weapon_model = YOLO(model_path)
                self.weapon_model.to(device)
                # Print the class names so user can see what the model detects
                names = self.weapon_model.names
                print(f"[INFO] Weapon model loaded ({device}): {model_path}")
                print(f"       Classes: {names}")
            except Exception as exc:
                print(f"[ERROR] Failed to load weapon model: {exc}")
        else:
            print(f"[WARNING] Weapon model not found: {model_path}")
            print("          Weapon detection disabled. Place best.pt in backend/models/")

    def detect(self, frame: np.ndarray) -> dict:
        """
        Run detection on a single BGR frame.

        Returns dict with:
            detections      : list[dict]
            people_count    : int
            weapon_detected : bool
        """
        detections = []
        people_count = 0
        weapon_detected = False

        # ── Person detection (yolov8n, COCO class 0 = "person") ──
        if self.person_model is not None:
            results = self.person_model.predict(
                source=frame, conf=self.conf_threshold,
                classes=[0],  # only detect person (COCO class 0)
                verbose=False,
            )
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    detections.append({
                        "label": "Person",
                        "confidence": round(conf, 2),
                        "box": (x1, y1, x2, y2),
                        "color": LABEL_COLOURS["Person"],
                    })
                    people_count += 1

        # ── Weapon detection (custom model) ──
        if self.weapon_model is not None:
            results = self.weapon_model.predict(
                source=frame, conf=self.conf_threshold, verbose=False,
            )
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    raw_label = result.names.get(cls_id, "unknown").lower()

                    # Skip person detections from weapon model (avoid duplicates)
                    if raw_label == "person":
                        continue

                    # Check if it's a weapon-type label
                    is_weapon = any(w in raw_label for w in WEAPON_LABELS)
                    if not is_weapon:
                        continue

                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

                    # Map to canonical label
                    if "gun" in raw_label or "pistol" in raw_label or "rifle" in raw_label:
                        canonical = "Gun"
                    else:
                        canonical = "Knife"

                    detections.append({
                        "label": canonical,
                        "confidence": round(conf, 2),
                        "box": (x1, y1, x2, y2),
                        "color": LABEL_COLOURS.get(canonical, DEFAULT_COLOUR),
                    })
                    weapon_detected = True

        return {
            "detections": detections,
            "people_count": people_count,
            "weapon_detected": weapon_detected,
        }

    @staticmethod
    def draw_detections(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        """Draw bounding boxes and labels onto a copy of the frame."""
        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            color = det["color"]
            label_text = f"{det['label']}  {det['confidence']:.0%}"
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cs = 14
            for cx, cy, dx, dy in [
                (x1, y1, 1, 1), (x2, y1, -1, 1),
                (x1, y2, 1, -1), (x2, y2, -1, -1),
            ]:
                cv2.line(out, (cx, cy), (cx + dx * cs, cy), color, 3)
                cv2.line(out, (cx, cy), (cx, cy + dy * cs), color, 3)
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, y1 - th - 12), (x1 + tw + 12, y1), color, -1)
            cv2.putText(out, label_text, (x1 + 6, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 1, cv2.LINE_AA)
        return out
