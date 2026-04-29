"""
alert.py - Alert system with snapshots and console messages.
"""

import os
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
    """Generates alerts with snapshots and console messages."""

    def __init__(self, alert_dir: str = ALERT_DIR, cooldown: float = ALERT_COOLDOWN_SECONDS):
        self.alert_dir = alert_dir
        self.cooldown = cooldown
        self._last_alert_time: float = 0.0
        self.alert_log: list[dict] = []
        os.makedirs(self.alert_dir, exist_ok=True)
        print(f"[INFO] Alert snapshots dir: {os.path.abspath(self.alert_dir)}")

    def trigger(self, frame: np.ndarray, threat_level: str, reason: str) -> dict | None:
        """Trigger alert if threat != SAFE and cooldown elapsed."""
        if threat_level == "SAFE":
            return None

        now = datetime.now()
        if (now.timestamp() - self._last_alert_time) < self.cooldown:
            return None

        self._last_alert_time = now.timestamp()
        ts_str = now.strftime("%H:%M:%S")
        ts_file = now.strftime("%Y%m%d_%H%M%S_%f")

        snapshot_path = os.path.join(self.alert_dir, f"{ts_file}.jpg")
        try:
            cv2.imwrite(snapshot_path, frame)
        except Exception as exc:
            print(f"[ERROR] Snapshot save failed: {exc}")
            snapshot_path = None

        colour = _COLOURS.get(threat_level, "")
        reset = _COLOURS["RESET"]
        print(f"{colour}[{ts_str}] {threat_level}: {reason}{reset}")

        record = {"timestamp": ts_str, "level": threat_level, "reason": reason, "snapshot": snapshot_path}
        self.alert_log.append(record)
        return record

    def get_recent_alerts(self, n: int = 10) -> list[dict]:
        return self.alert_log[-n:]

    def clear(self):
        self.alert_log.clear()
        self._last_alert_time = 0.0
