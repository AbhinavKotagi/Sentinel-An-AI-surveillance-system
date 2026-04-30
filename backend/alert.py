"""
alert.py - Alert system with snapshots, console messages, and feedback integration.
"""

import os
import uuid
import cv2
import numpy as np
from datetime import datetime

ALERT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "alerts")
ALERT_COOLDOWN_SECONDS = 3.0

_COLOURS = {
    "CRITICAL": "\033[1;91m",
    "HIGH":     "\033[1;93m",
    "MEDIUM":   "\033[1;33m",
    "RESET":    "\033[0m",
}


class AlertSystem:
    """Generates alerts with snapshots, console messages, and unique IDs for feedback tracking."""

    def __init__(self, alert_dir: str = ALERT_DIR, cooldown: float = ALERT_COOLDOWN_SECONDS):
        self.alert_dir = alert_dir
        self.cooldown = cooldown
        self._last_alert_time: float = 0.0
        self.alert_log: list[dict] = []
        os.makedirs(self.alert_dir, exist_ok=True)
        print(f"[INFO] Alert snapshots dir: {os.path.abspath(self.alert_dir)}")

    def trigger(
        self,
        frame: np.ndarray,
        threat_level: str,
        reason: str,
        detection_type: str = "unknown",
        signals: dict | None = None,
        log_safe: bool = False,
    ) -> dict | None:
        """
        Trigger alert if threat != SAFE and cooldown elapsed.

        Parameters
        ----------
        frame : np.ndarray
            Current video frame for snapshot.
        threat_level : str
            SAFE, MEDIUM, HIGH, or CRITICAL.
        reason : str
            Human-readable reason for the alert.
        detection_type : str
            Category: 'weapon', 'fight', 'combined'.
        signals : dict | None
            Raw detection signals for ML feedback (YOLO conf, fight score, etc.).
        log_safe : bool
            If True, allows logging SAFE frames for review (missed threats).

        Returns
        -------
        dict | None
            Alert record with unique ID, or None if suppressed/safe.
        """
        if threat_level == "SAFE" and not log_safe:
            return None

        now = datetime.now()
        if (now.timestamp() - self._last_alert_time) < self.cooldown:
            return None

        self._last_alert_time = now.timestamp()
        ts_str = now.strftime("%H:%M:%S")
        ts_iso = now.isoformat()
        ts_file = now.strftime("%Y%m%d_%H%M%S_%f")

        # Generate unique alert ID
        alert_id = str(uuid.uuid4())[:12]

        snapshot_path = os.path.join(self.alert_dir, f"{ts_file}.jpg")
        try:
            cv2.imwrite(snapshot_path, frame)
        except Exception as exc:
            print(f"[ERROR] Snapshot save failed: {exc}")
            snapshot_path = None

        colour = _COLOURS.get(threat_level, "")
        reset = _COLOURS["RESET"]
        print(f"{colour}[{ts_str}] {threat_level}: {reason}{reset}")

        record = {
            "alert_id": alert_id,
            "timestamp": ts_str,
            "timestamp_iso": ts_iso,
            "level": threat_level,
            "reason": reason,
            "snapshot": snapshot_path,
            "detection_type": detection_type,
            "signals": signals or {},
        }
        self.alert_log.append(record)
        return record

    def get_recent_alerts(self, n: int = 10) -> list[dict]:
        return self.alert_log[-n:]

    def clear(self):
        self.alert_log.clear()
        self._last_alert_time = 0.0
