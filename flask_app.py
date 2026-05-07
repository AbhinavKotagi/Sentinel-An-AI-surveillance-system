"""
flask_app.py — Flask backend for SENTINEL AI Surveillance Dashboard.

Provides:
  /             → Serves the HTML dashboard
  /video_feed   → Live MJPEG video stream with AI overlays
  /status       → JSON endpoint for real-time detection data
  /alerts       → JSON endpoint for alert history

Integrates the existing detection pipeline:
  YOLO → MediaPipe → Motion → Fight logic → Threat level
Falls back to simulated data if AI modules fail to load.
"""

import sys
import os
import time
import json
import random
import threading
from datetime import datetime

# Ensure backend modules are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

import cv2
import numpy as np

from flask import Flask, Response, jsonify, send_from_directory
from flask_cors import CORS

# ─── Try importing AI modules (graceful fallback) ────────────────────────────
AI_AVAILABLE = False
try:
    from detector import ThreatDetector
    from pose import PoseEstimator
    from motion import MotionDetector
    from logic import FightDetector, compute_threat_level, classify_detection_type
    from alert import AlertSystem
    AI_AVAILABLE = True
    print("[INFO] All AI modules loaded successfully.")
except ImportError as e:
    print(f"[WARNING] AI modules not available: {e}")
    print("[INFO] Running in SIMULATION mode with random data.")

# ─── Flask App ────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

# ─── Thread-safe shared state ────────────────────────────────────────────────
state_lock = threading.Lock()
shared_state = {
    "threat": "SAFE",
    "persons": 0,
    "weapon": False,
    "motion": 0,
    "movement": 0,
    "fight_score": 0.0,
    "arm_speed": 0.0,
    "leg_speed": 0.0,
    "proximity": 0.0,
    "fight_detected": False,
    "reason": "",
    "ai_confidence": 0.0,
    "objects_detected": 0,
    "fps": 0.0,
}

# Alert history (thread-safe)
alerts_lock = threading.Lock()
alert_history = []
MAX_ALERTS = 50

# Latest frame for streaming
frame_lock = threading.Lock()
latest_frame = None

# ─── AI Pipeline (initialised once) ──────────────────────────────────────────
detector = None
pose_est = None
motion_det = None
fight_det = None
alert_sys = None

if AI_AVAILABLE:
    try:
        detector = ThreatDetector(conf_threshold=0.40)
    except Exception as e:
        print(f"[ERROR] Failed to init ThreatDetector: {e}")
    try:
        pose_est = PoseEstimator()
    except Exception as e:
        print(f"[ERROR] Failed to init PoseEstimator: {e}")
    try:
        motion_det = MotionDetector()
    except Exception as e:
        print(f"[ERROR] Failed to init MotionDetector: {e}")
    try:
        fight_det = FightDetector()
    except Exception as e:
        print(f"[ERROR] Failed to init FightDetector: {e}")
    try:
        alert_sys = AlertSystem()
    except Exception as e:
        print(f"[ERROR] Failed to init AlertSystem: {e}")


# ─── Draw overlays on frame ──────────────────────────────────────────────────
def draw_overlays(frame, detections, threat_level):
    """Draw detection boxes and threat indicator on frame."""
    out = frame.copy()
    h, w = out.shape[:2]
    tc_map = {
        "SAFE": (0, 255, 163), "MEDIUM": (255, 224, 80),
        "HIGH": (255, 140, 0), "CRITICAL": (255, 34, 85),
    }
    tc = tc_map.get(threat_level, (200, 200, 200))

    for obj in detections:
        x1, y1, x2, y2 = obj["box"]
        c = obj["color"]
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 1)
        cs = 10
        for cx, cy, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                                (x1, y2, 1, -1), (x2, y2, -1, -1)]:
            cv2.line(out, (cx, cy), (cx + dx * cs, cy), c, 2)
            cv2.line(out, (cx, cy), (cx, cy + dy * cs), c, 2)
        label = f"{obj['label']}  {obj['confidence']:.0%}"
        (tw, th_), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(out, (x1, y1 - th_ - 8), (x1 + tw + 8, y1), c, -1)
        cv2.putText(out, label, (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (5, 10, 15), 1, cv2.LINE_AA)

    # Threat level badge (top-right)
    tl_text = threat_level
    (ttw, tth), _ = cv2.getTextSize(tl_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    tx = w - ttw - 14
    cv2.rectangle(out, (tx - 6, 6), (w - 6, tth + 16), tc, -1)
    cv2.putText(out, tl_text, (tx, tth + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (5, 10, 15), 1, cv2.LINE_AA)

    # Timestamp (bottom-left)
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(out, ts, (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60, 100, 130), 1, cv2.LINE_AA)

    # Critical red border
    if threat_level == "CRITICAL":
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), (255, 34, 85), 3)

    return out


# ─── Simulation fallback ─────────────────────────────────────────────────────
_sim_prev_threat = "SAFE"


def simulate_status():
    """Generate realistic simulated detection data."""
    global _sim_prev_threat
    threats = ["SAFE", "SAFE", "SAFE", "MEDIUM", "HIGH", "CRITICAL"]
    threat = random.choice(threats)
    persons = random.randint(0, 5)
    weapon = threat in ("HIGH", "CRITICAL") and random.random() > 0.5
    motion = random.randint(5, 95)
    movement = random.randint(5, 80)

    # Generate alert on threat change (not SAFE)
    if threat != "SAFE" and threat != _sim_prev_threat:
        reason_map = {
            "MEDIUM": "Elevated motion in monitored zone",
            "HIGH": "Aggressive motion + proximity detected",
            "CRITICAL": f"Weapon detected ({random.randint(80, 98)}% confidence)",
        }
        add_alert(threat, reason_map.get(threat, "Anomaly detected"))

    _sim_prev_threat = threat

    return {
        "threat": threat,
        "persons": persons,
        "weapon": weapon,
        "motion": motion,
        "movement": movement,
        "fight_score": round(random.uniform(0, 1), 3),
        "arm_speed": round(random.uniform(0, 0.8), 3),
        "leg_speed": round(random.uniform(0, 0.6), 3),
        "proximity": round(random.uniform(0, 1), 3),
        "fight_detected": threat in ("MEDIUM", "HIGH"),
        "reason": "",
        "ai_confidence": round(random.uniform(60, 99), 1),
        "objects_detected": persons + random.randint(0, 8),
        "fps": round(random.uniform(24, 30), 1),
    }


def add_alert(level, message):
    """Thread-safe alert insertion."""
    with alerts_lock:
        alert_history.insert(0, {
            "level": level,
            "message": message,
            "time": datetime.now().strftime("%H:%M:%S"),
            "timestamp": datetime.now().isoformat(),
        })
        # Keep only the latest MAX_ALERTS
        while len(alert_history) > MAX_ALERTS:
            alert_history.pop()


# ─── Video capture + processing thread ────────────────────────────────────────
_prev_threat_for_alert = "SAFE"


def video_processing_thread():
    """Background thread: capture → detect → update shared state + frame."""
    global latest_frame, _prev_threat_for_alert

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Video feed will not be available.")
        return

    # Optimise webcam
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_count = 0
    prev_det = None
    prev_pose = None
    fps_timer = time.time()
    fps_count = 0
    current_fps = 0.0

    print("[INFO] Video processing thread started.")

    while True:
        # Grab extra frame to drop stale buffer
        cap.grab()
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        frame_count += 1
        fps_count += 1

        # FPS calculation
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            current_fps = fps_count / elapsed
            fps_count = 0
            fps_timer = time.time()

        # Resize for processing
        frame = cv2.resize(frame, (640, 480))

        if AI_AVAILABLE and detector is not None:
            # ── Real AI Pipeline ──
            # YOLO detection (every frame for responsiveness)
            if frame_count % 2 == 0 or prev_det is None:
                det_result = detector.detect(frame)
                prev_det = det_result
            else:
                det_result = prev_det

            detections = det_result["detections"]
            people_count = det_result["people_count"]
            weapon_detected = det_result["weapon_detected"]

            # MediaPipe pose
            if pose_est is not None:
                if frame_count % 2 == 0 or prev_pose is None:
                    pose_result = pose_est.process(frame)
                    prev_pose = pose_result
                    landmarks = pose_result["landmarks"]
                    frame = pose_result["frame"]
                else:
                    landmarks = prev_pose["landmarks"]
                    frame = prev_pose["frame"]
            else:
                landmarks = None

            # Motion detection
            if motion_det is not None:
                small = cv2.resize(frame, (320, 240))
                motion_score = motion_det.compute(small)
            else:
                motion_score = 0.0

            # Fight detection
            if fight_det is not None:
                fight_result = fight_det.detect_fight(landmarks, detections, motion_score)
                fight_detected = fight_result["fight_detected"]
                fight_reason = fight_result.get("reason", "")
            else:
                fight_result = {"fight_score": 0.0, "arm_speed": 0.0,
                                "leg_speed": 0.0, "proximity": 0.0,
                                "fight_detected": False, "movement_score": 0.0}
                fight_detected = False
                fight_reason = ""

            # Threat level
            threat_level, reason = compute_threat_level(
                weapon_detected, fight_detected, people_count, fight_reason
            )

            # Max detection confidence
            max_conf = max((d["confidence"] for d in detections), default=0.0)

            # Draw overlays
            processed_frame = draw_overlays(frame, detections, threat_level)

            # Generate alerts on threat transitions
            if threat_level != "SAFE" and threat_level != _prev_threat_for_alert:
                alert_msg = reason if reason else f"Threat level changed to {threat_level}"
                add_alert(threat_level, alert_msg)
            _prev_threat_for_alert = threat_level

            # Update shared state
            with state_lock:
                shared_state.update({
                    "threat": threat_level,
                    "persons": people_count,
                    "weapon": weapon_detected,
                    "motion": int(motion_score * 100),
                    "movement": int(fight_result.get("movement_score", 0) * 100),
                    "fight_score": fight_result.get("fight_score", 0.0),
                    "arm_speed": fight_result.get("arm_speed", 0.0),
                    "leg_speed": fight_result.get("leg_speed", 0.0),
                    "proximity": fight_result.get("proximity", 0.0),
                    "fight_detected": fight_detected,
                    "reason": reason,
                    "ai_confidence": round(max_conf * 100, 1),
                    "objects_detected": len(detections),
                    "fps": round(current_fps, 1),
                })
        else:
            # ── Simulation mode: just stream raw webcam ──
            processed_frame = frame

            # Update with simulated data periodically
            if frame_count % 30 == 0:
                sim_data = simulate_status()
                with state_lock:
                    shared_state.update(sim_data)
                    shared_state["fps"] = round(current_fps, 1)

        # Encode frame as JPEG
        _, buffer = cv2.imencode(".jpg", processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with frame_lock:
            latest_frame = buffer.tobytes()

        # Target ~30 fps
        time.sleep(0.033)

    cap.release()


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def serve_index():
    """Serve the main HTML dashboard."""
    return send_from_directory(".", "index.html")


@app.route("/script.js")
def serve_script():
    """Serve the JavaScript file."""
    return send_from_directory(".", "script.js")


def generate_frames():
    """Generator: yield MJPEG frames for streaming."""
    while True:
        with frame_lock:
            frame_bytes = latest_frame
        if frame_bytes is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
        else:
            # No frame yet — yield a blank
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "CONNECTING...", (180, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
            _, buf = cv2.imencode(".jpg", blank)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            )
        time.sleep(0.033)  # ~30 fps


@app.route("/video_feed")
def video_feed():
    """Live MJPEG video stream."""
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/status")
def status():
    """Return current detection data as JSON."""
    with state_lock:
        data = dict(shared_state)
    return jsonify(data)


@app.route("/alerts")
def alerts():
    """Return alert history as JSON."""
    with alerts_lock:
        data = list(alert_history)
    return jsonify(data)


# ─── Start ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Start video processing in background thread
    video_thread = threading.Thread(target=video_processing_thread, daemon=True)
    video_thread.start()

    print("\n" + "=" * 60)
    print("  SENTINEL — AI Surveillance Dashboard")
    print("  Server running at: http://localhost:5000")
    print("  Video feed at:     http://localhost:5000/video_feed")
    print("  Status API at:     http://localhost:5000/status")
    print("=" * 60 + "\n")

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
