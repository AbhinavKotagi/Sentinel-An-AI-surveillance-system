"""
main.py - Real-Time AI Threat Detection System — main execution loop.

Pipeline per frame:
  1. Capture & resize frame  (threaded, non-blocking)
  2. YOLO detection (people, weapons)  — every 3rd frame
  3. MediaPipe pose estimation (skeleton) — every 3rd frame
  4. Motion detection (frame differencing) — on downscaled frame
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

from camera_thread import CameraThread
from detector import ThreatDetector
from pose import PoseEstimator
from motion import MotionDetector
from logic import FightDetector, compute_threat_level
from alert import AlertSystem


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
# [OPTIMIZATION] Reduced processing resolution for faster inference
PROCESS_WIDTH  = 480
PROCESS_HEIGHT = 360
WINDOW_NAME    = "SENTINEL — AI Threat Detection"

# [OPTIMIZATION] Even smaller resolution for motion detection only
MOTION_WIDTH  = 320
MOTION_HEIGHT = 240

# [OPTIMIZATION] Frame-skip intervals (run heavy models less often)
YOLO_SKIP_INTERVAL = 3   # Run YOLO every 3rd frame
POSE_SKIP_INTERVAL = 3   # Run MediaPipe every 3rd frame

# Threat colour map (BGR)
THREAT_COLOURS = {
    "SAFE":     (0, 255, 163),
    "MEDIUM":   (0, 224, 255),
    "HIGH":     (0, 140, 255),
    "CRITICAL": (85, 34, 255),
}


# ──────────────────────────────────────────────────────────────────────────────
# HUD overlay drawing  [OPTIMIZATION] Reduced font sizes & line thickness
# ──────────────────────────────────────────────────────────────────────────────
def draw_hud(frame: np.ndarray, threat_level: str, frame_data: dict) -> np.ndarray:
    """Draw threat badge, stats, and timestamp onto the frame."""
    out = frame.copy()
    h, w = out.shape[:2]
    tc = THREAT_COLOURS.get(threat_level, (200, 200, 200))

    # ── Threat level badge (top-right) — smaller font ──
    label = threat_level
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    tx = w - tw - 16
    cv2.rectangle(out, (tx - 8, 4), (w - 4, th + 18), tc, -1)
    cv2.putText(out, label, (tx, th + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 1, cv2.LINE_AA)

    # ── Stats bar (top-left) — smaller font ──
    stats_lines = [
        f"People: {frame_data['people']}",
        f"Weapon: {'YES' if frame_data['weapon'] else 'NO'}",
        f"Motion: {frame_data['motion_score']:.0%}",
        f"Fight:  {'YES' if frame_data['fight'] else 'NO'}",
    ]
    y0 = 20
    for i, line in enumerate(stats_lines):
        cv2.putText(out, line, (10, y0 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 220, 240), 1, cv2.LINE_AA)

    # ── Timestamp (bottom-left) ──
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(out, ts, (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 120, 150), 1, cv2.LINE_AA)

    # ── Red border flash for CRITICAL — thinner border ──
    if threat_level == "CRITICAL":
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 0, 255), 3)

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SENTINEL — AI Threat Detection System")
    parser.add_argument("--source", default="0",
                        help="Video source: webcam index (0), path to video file, "
                             "or WiFi/IP camera URL (e.g. http://192.168.1.5:8080/video)")
    parser.add_argument("--conf", type=float, default=0.40,
                        help="YOLO confidence threshold (default: 0.40)")
    args = parser.parse_args()

    # Determine source — keep URLs as strings, convert digits to int
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

    # [OPTIMIZATION] Use threaded camera for non-blocking capture
    print(f"[INFO] Opening video source: {source}")
    cam = CameraThread(source=source, width=PROCESS_WIDTH, height=PROCESS_HEIGHT, fps=30)

    if not cam.is_opened():
        print(f"[ERROR] Cannot open video source: {source}")
        sys.exit(1)

    cam.start()
    print("[INFO] Threaded camera started.")
    print("[INFO] Press 'q' to quit.\n")

    fps_timer = time.time()
    fps_count = 0
    display_fps = 0.0

    # [OPTIMIZATION] Frame counter & cached results for frame-skipping
    frame_count = 0
    prev_det_result = None
    prev_pose_result = None

    try:
        while True:
            frame_count += 1

            # [OPTIMIZATION] Non-blocking read from threaded camera
            ret, frame = cam.read()
            if not ret or frame is None:
                if isinstance(source, str) and not source.isdigit():
                    # Loop video files
                    cam.reset()
                    time.sleep(0.01)
                    continue
                # No frame yet on first iteration — wait briefly
                if frame_count <= 5:
                    time.sleep(0.05)
                    continue
                print("[INFO] Video stream ended.")
                break

            # [OPTIMIZATION] Resize to processing resolution early,
            # BEFORE any detection/pose/motion computation
            frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))

            # ── 1. YOLO Detection ──
            # [OPTIMIZATION] Run YOLO every 3rd frame, reuse cached result otherwise
            if frame_count % YOLO_SKIP_INTERVAL == 0 or prev_det_result is None:
                det_result = detector.detect(frame)
                prev_det_result = det_result
            else:
                det_result = prev_det_result

            detections     = det_result["detections"]
            people_count   = det_result["people_count"]
            weapon_detected = det_result["weapon_detected"]

            # ── 2. Pose Estimation ──
            # [DISABLED] MediaPipe pose estimation — commented out for performance
            # if frame_count % POSE_SKIP_INTERVAL == 0 or prev_pose_result is None:
            #     pose_result = pose_est.process(frame)
            #     prev_pose_result = pose_result
            #     landmarks   = pose_result["landmarks"]
            #     frame       = pose_result["frame"]
            # else:
            #     pose_result = prev_pose_result
            #     landmarks   = pose_result["landmarks"]
            landmarks = None

            # ── 3. Motion Detection ──
            # [OPTIMIZATION] Downscale to 320×240 for motion only
            small_frame = cv2.resize(frame, (MOTION_WIDTH, MOTION_HEIGHT))
            motion_score = motion_det.compute(small_frame)

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
            if threat_level != "SAFE":
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
            cv2.putText(frame, f"FPS: {display_fps:.0f}", (PROCESS_WIDTH - 100, PROCESS_HEIGHT - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 120, 150), 1, cv2.LINE_AA)

            # [OPTIMIZATION] Always display latest frame for smooth appearance
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # q or Esc
                print("[INFO] Quitting...")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        cam.stop()
        cv2.destroyAllWindows()
        pose_est.release()
        print("[INFO] Resources released. Goodbye.")


if __name__ == "__main__":
    main()
