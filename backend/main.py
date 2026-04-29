"""
main.py - Real-Time AI Threat Detection System — main execution loop.

Pipeline per frame:
  1. Capture & resize frame
  2. YOLO detection (people, weapons)
  3. MediaPipe pose estimation (skeleton)
  4. Motion detection (frame differencing)
  5. Fight detection (arm speed + proximity + motion)
  6. Threat level computation
  7. Alert generation (snapshot + console)
  8. Overlay drawing & display

Usage:
  python main.py                    # webcam (default)
  python main.py --source video.mp4 # video file
  python main.py --source 0         # explicit webcam index
"""

import argparse
import sys
import time
import cv2
import numpy as np
from datetime import datetime

from detector import ThreatDetector
from pose import PoseEstimator
from motion import MotionDetector
from logic import FightDetector, compute_threat_level
from alert import AlertSystem


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
WINDOW_NAME  = "SENTINEL — AI Threat Detection"

# Threat colour map (BGR)
THREAT_COLOURS = {
    "SAFE":     (0, 255, 163),
    "MEDIUM":   (0, 224, 255),
    "HIGH":     (0, 140, 255),
    "CRITICAL": (85, 34, 255),
}


# ──────────────────────────────────────────────────────────────────────────────
# HUD overlay drawing
# ──────────────────────────────────────────────────────────────────────────────
def draw_hud(frame: np.ndarray, threat_level: str, frame_data: dict) -> np.ndarray:
    """Draw threat badge, stats, and timestamp onto the frame."""
    out = frame.copy()
    h, w = out.shape[:2]
    tc = THREAT_COLOURS.get(threat_level, (200, 200, 200))

    # ── Threat level badge (top-right) ──
    label = threat_level
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    tx = w - tw - 20
    cv2.rectangle(out, (tx - 10, 6), (w - 6, th + 22), tc, -1)
    cv2.putText(out, label, (tx, th + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 10, 10), 2, cv2.LINE_AA)

    # ── Stats bar (top-left) ──
    stats_lines = [
        f"People: {frame_data['people']}",
        f"Weapon: {'YES' if frame_data['weapon'] else 'NO'}",
        f"Motion: {frame_data['motion_score']:.0%}",
        f"Fight:  {'YES' if frame_data['fight'] else 'NO'}",
    ]
    y0 = 24
    for i, line in enumerate(stats_lines):
        cv2.putText(out, line, (12, y0 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 220, 240), 1, cv2.LINE_AA)

    # ── Timestamp (bottom-left) ──
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(out, ts, (10, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 120, 150), 1, cv2.LINE_AA)

    # ── Red border flash for CRITICAL ──
    if threat_level == "CRITICAL":
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SENTINEL — AI Threat Detection System")
    parser.add_argument("--source", default="0",
                        help="Video source: webcam index (0) or path to video file")
    parser.add_argument("--conf", type=float, default=0.40,
                        help="YOLO confidence threshold (default: 0.40)")
    args = parser.parse_args()

    # Determine source
    source = int(args.source) if args.source.isdigit() else args.source

    # ── Initialise modules ──
    print("=" * 60)
    print("  SENTINEL — Real-Time AI Threat Detection System")
    print("=" * 60)

    detector     = ThreatDetector(conf_threshold=args.conf)
    pose_est     = PoseEstimator()
    motion_det   = MotionDetector()
    fight_det    = FightDetector()
    alert_system = AlertSystem()

    print(f"[INFO] Opening video source: {source}")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        sys.exit(1)

    print("[INFO] Press 'q' to quit.\n")

    fps_timer = time.time()
    fps_count = 0
    display_fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if isinstance(source, str) and not source.isdigit():
                    # Loop video files
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                print("[INFO] Video stream ended.")
                break

            # ── Resize for performance ──
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            # ── 1. YOLO Detection ──
            det_result = detector.detect(frame)
            detections     = det_result["detections"]
            people_count   = det_result["people_count"]
            weapon_detected = det_result["weapon_detected"]

            # ── 2. Pose Estimation ──
            pose_result = pose_est.process(frame)
            landmarks   = pose_result["landmarks"]
            frame       = pose_result["frame"]  # frame now has skeleton overlay

            # ── 3. Motion Detection ──
            motion_score = motion_det.compute(frame)

            # ── 4. Fight Detection ──
            fight_result = fight_det.detect_fight(landmarks, detections, motion_score)
            fight_detected = fight_result["fight_detected"]

            # ── 5. Threat Level ──
            threat_level, reason = compute_threat_level(weapon_detected, fight_detected)

            # ── 6. Build frame data dict (for frontend) ──
            frame_data = {
                "people":       people_count,
                "weapon":       weapon_detected,
                "motion_score": round(motion_score, 3),
                "fight":        fight_detected,
                "threat":       threat_level,
            }

            # ── 7. Alert ──
            alert_system.trigger(frame, threat_level, reason)

            # ── 8. Draw overlays ──
            frame = ThreatDetector.draw_detections(frame, detections)
            frame = draw_hud(frame, threat_level, frame_data)

            # ── FPS counter ──
            fps_count += 1
            if time.time() - fps_timer >= 1.0:
                display_fps = fps_count
                fps_count = 0
                fps_timer = time.time()
            cv2.putText(frame, f"FPS: {display_fps:.0f}", (FRAME_WIDTH - 110, FRAME_HEIGHT - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 120, 150), 1, cv2.LINE_AA)

            # ── Display ──
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # q or Esc
                print("[INFO] Quitting...")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose_est.release()
        print("[INFO] Resources released. Goodbye.")


if __name__ == "__main__":
    main()
