"""
feedback.py — Threat Evaluation & ML Feedback Engine.

Three components:
  1. FeedbackStore     — SQLite-backed storage for alerts + operator verdicts
  2. ThreatMetaClassifier — GradientBoosting meta-classifier that learns to
                            distinguish true threats from false alarms
  3. ParameterOptimizer   — Grid-search calibrator that finds optimal values
                            for YOLO conf, motion sensitivity, fight threshold

The meta-classifier is trained on operator feedback and runs inference on
every new alert to suppress likely false positives.  The parameter optimizer
replays labelled data to find UI slider values that maximise F1 score.
"""

import os
import json
import sqlite3
import uuid
import threading
from datetime import datetime

import numpy as np

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    import joblib
    _ML_AVAILABLE = True
except ImportError:
    _ML_AVAILABLE = False
    print("[WARNING] scikit-learn not installed. Run: pip install scikit-learn joblib")


# ──────────────────────────────────────────────────────────────────────────────
# Default paths
# ──────────────────────────────────────────────────────────────────────────────
_BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "feedback")
DEFAULT_DB_PATH = os.path.join(_BASE_DIR, "feedback.db")
DEFAULT_MODEL_PATH = os.path.join(_BASE_DIR, "meta_model.joblib")

# Feature columns used by the meta-classifier (order matters)
FEATURE_COLS = [
    "yolo_confidence",
    "people_count",
    "weapon_detected",
    "motion_score",
    "arm_speed",
    "leg_speed",
    "proximity",
    "fight_score",
    "threat_level_enc",
    "hour_of_day",
]

THREAT_LEVEL_MAP = {"SAFE": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


# ══════════════════════════════════════════════════════════════════════════════
# 1. FeedbackStore — SQLite storage
# ══════════════════════════════════════════════════════════════════════════════

class FeedbackStore:
    """SQLite-backed storage for alert records and operator feedback."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._persistent_conn = None  # used for :memory: databases
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    def _get_conn(self) -> sqlite3.Connection:
        if self.db_path == ":memory:":
            if self._persistent_conn is None:
                self._persistent_conn = sqlite3.connect(":memory:", check_same_thread=False)
                self._persistent_conn.row_factory = sqlite3.Row
            return self._persistent_conn
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _close_conn(self, conn):
        """Close connection (no-op for :memory: databases)."""
        if self.db_path != ":memory:":
            conn.close()

    def _init_db(self):
        with self._lock:
            conn = self._get_conn()
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id              TEXT PRIMARY KEY,
                    timestamp       TEXT NOT NULL,
                    threat_level    TEXT NOT NULL,
                    reason          TEXT,
                    snapshot        TEXT,
                    det_type        TEXT,
                    signals         TEXT,
                    verdict         TEXT,
                    notes           TEXT,
                    reviewed_at     TEXT,
                    ml_prediction   REAL
                );

                CREATE TABLE IF NOT EXISTS model_history (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    trained_at          TEXT NOT NULL,
                    n_samples           INTEGER,
                    accuracy            REAL,
                    precision_score     REAL,
                    recall              REAL,
                    f1                  REAL,
                    feature_importances TEXT
                );

                CREATE TABLE IF NOT EXISTS calibration_history (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    calibrated_at   TEXT NOT NULL,
                    yolo_conf       REAL,
                    motion_sens     REAL,
                    fight_thresh    REAL,
                    f1_score        REAL
                );
            """)
            conn.commit()
            self._close_conn(conn)

    # ------------------------------------------------------------------
    # Record & feedback
    # ------------------------------------------------------------------
    def record_alert(
        self,
        alert_id: str,
        timestamp: str,
        threat_level: str,
        reason: str,
        snapshot_path: str | None,
        det_type: str,
        signals: dict,
        ml_prediction: float | None = None,
    ) -> None:
        """Log a triggered alert with its raw detection signals."""
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """INSERT OR REPLACE INTO alerts
                   (id, timestamp, threat_level, reason, snapshot, det_type,
                    signals, ml_prediction)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (alert_id, timestamp, threat_level, reason, snapshot_path,
                 det_type, json.dumps(signals), ml_prediction),
            )
            conn.commit()
            self._close_conn(conn)

    def submit_feedback(self, alert_id: str, verdict: str, notes: str = "") -> None:
        """Operator marks an alert as 'true_positive' or 'false_positive'."""
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """UPDATE alerts SET verdict = ?, notes = ?, reviewed_at = ?
                   WHERE id = ?""",
                (verdict, notes, datetime.now().isoformat(), alert_id),
            )
            conn.commit()
            self._close_conn(conn)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get_pending_reviews(self, limit: int = 20) -> list[dict]:
        """Get alerts that haven't been reviewed yet."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM alerts WHERE verdict IS NULL ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        self._close_conn(conn)
        return [self._row_to_dict(r) for r in rows]

    def get_reviewed(self, limit: int = 100) -> list[dict]:
        """Get reviewed alerts, most recent first."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM alerts WHERE verdict IS NOT NULL ORDER BY reviewed_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        self._close_conn(conn)
        return [self._row_to_dict(r) for r in rows]

    def get_all_labelled(self) -> list[dict]:
        """Get all alerts that have operator verdicts (for ML training)."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM alerts WHERE verdict IS NOT NULL ORDER BY timestamp",
        ).fetchall()
        self._close_conn(conn)
        return [self._row_to_dict(r) for r in rows]

    def count_labelled(self) -> int:
        """Quick count of labelled samples."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) FROM alerts WHERE verdict IS NOT NULL"
        ).fetchone()
        self._close_conn(conn)
        return row[0] if row else 0

    def count_total(self) -> int:
        """Total number of alerts recorded."""
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()
        self._close_conn(conn)
        return row[0] if row else 0

    def get_stats(self) -> dict:
        """Compute accuracy statistics from labelled data."""
        labelled = self.get_all_labelled()
        if not labelled:
            return {
                "total_alerts": self.count_total(),
                "labelled": 0, "pending": 0,
                "true_positives": 0, "false_positives": 0,
                "precision": None, "by_category": {},
            }

        tp = sum(1 for a in labelled if a["verdict"] == "true_positive")
        fp = sum(1 for a in labelled if a["verdict"] == "false_positive")
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # Per-category
        categories = {}
        for a in labelled:
            cat = a.get("det_type", "unknown")
            if cat not in categories:
                categories[cat] = {"tp": 0, "fp": 0}
            if a["verdict"] == "true_positive":
                categories[cat]["tp"] += 1
            else:
                categories[cat]["fp"] += 1

        for cat in categories:
            t = categories[cat]["tp"]
            f = categories[cat]["fp"]
            categories[cat]["precision"] = t / (t + f) if (t + f) > 0 else 0.0

        return {
            "total_alerts": self.count_total(),
            "labelled": len(labelled),
            "pending": self.count_total() - len(labelled),
            "true_positives": tp,
            "false_positives": fp,
            "precision": round(precision, 3),
            "by_category": categories,
        }

    # ------------------------------------------------------------------
    # Model history
    # ------------------------------------------------------------------
    def record_training(self, metrics: dict) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO model_history
               (trained_at, n_samples, accuracy, precision_score, recall, f1,
                feature_importances)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (datetime.now().isoformat(), metrics.get("n_samples"),
             metrics.get("accuracy"), metrics.get("precision"),
             metrics.get("recall"), metrics.get("f1"),
             json.dumps(metrics.get("feature_importances", {}))),
        )
        conn.commit()
        self._close_conn(conn)

    def get_training_history(self, limit: int = 20) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM model_history ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        self._close_conn(conn)
        return [dict(r) for r in rows]

    def record_calibration(self, result: dict) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO calibration_history
               (calibrated_at, yolo_conf, motion_sens, fight_thresh, f1_score)
               VALUES (?, ?, ?, ?, ?)""",
            (datetime.now().isoformat(), result.get("yolo_conf"),
             result.get("motion_sensitivity"), result.get("fight_threshold"),
             result.get("f1_score")),
        )
        conn.commit()
        self._close_conn(conn)

    # ------------------------------------------------------------------
    @staticmethod
    def _row_to_dict(row) -> dict:
        d = dict(row)
        if d.get("signals"):
            try:
                d["signals"] = json.loads(d["signals"])
            except (json.JSONDecodeError, TypeError):
                d["signals"] = {}
        return d


# ══════════════════════════════════════════════════════════════════════════════
# 2. ThreatMetaClassifier — ML false-alarm filter
# ══════════════════════════════════════════════════════════════════════════════

class ThreatMetaClassifier:
    """
    GradientBoosting meta-classifier that learns from operator feedback
    to distinguish true threats from false alarms.

    Features are extracted from the detection signal vector.
    Trained incrementally as new feedback arrives.
    """

    MIN_SAMPLES = 30          # Don't train until we have this many labels
    RETRAIN_INTERVAL = 10     # Retrain every N new labelled samples
    SUPPRESSION_THRESHOLD = 0.40  # Suppress alerts below this confidence

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self._last_trained_count = 0
        self._load_model()

    def _load_model(self):
        """Try to load a previously saved model."""
        if not _ML_AVAILABLE:
            return
        if os.path.isfile(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                print(f"[INFO] Meta-classifier loaded from {self.model_path}")
            except Exception as exc:
                print(f"[WARNING] Could not load meta-classifier: {exc}")
                self.model = None

    def _save_model(self):
        """Persist model to disk."""
        if self.model is not None:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self.model, self.model_path)

    # ------------------------------------------------------------------
    def is_active(self) -> bool:
        """Whether the model has been trained and is ready for inference."""
        return self.model is not None

    def should_retrain(self, store: FeedbackStore) -> bool:
        """Check if enough new feedback has arrived to warrant retraining."""
        current_count = store.count_labelled()
        if current_count < self.MIN_SAMPLES:
            return False
        return (current_count - self._last_trained_count) >= self.RETRAIN_INTERVAL

    # ------------------------------------------------------------------
    def extract_features(self, signals: dict) -> np.ndarray:
        """Convert a signals dict to a feature vector."""
        features = []
        for col in FEATURE_COLS:
            val = signals.get(col, 0)
            if isinstance(val, bool):
                val = int(val)
            elif col == "threat_level_enc" and isinstance(val, str):
                val = THREAT_LEVEL_MAP.get(val, 0)
            features.append(float(val) if val is not None else 0.0)
        return np.array(features).reshape(1, -1)

    def _prepare_dataset(self, labelled_data: list[dict]):
        """Build X, y arrays from labelled alert records."""
        X_list = []
        y_list = []
        for alert in labelled_data:
            signals = alert.get("signals", {})
            if not signals:
                continue

            # Ensure threat_level_enc is populated
            if "threat_level_enc" not in signals:
                signals["threat_level_enc"] = THREAT_LEVEL_MAP.get(
                    alert.get("threat_level", "SAFE"), 0
                )
            # Ensure hour_of_day
            if "hour_of_day" not in signals:
                try:
                    ts = alert.get("timestamp", "")
                    if "T" in ts:
                        signals["hour_of_day"] = int(ts.split("T")[1][:2])
                    elif ":" in ts:
                        signals["hour_of_day"] = int(ts.split(":")[0])
                    else:
                        signals["hour_of_day"] = 12
                except (ValueError, IndexError):
                    signals["hour_of_day"] = 12

            feat = self.extract_features(signals)
            label = 1 if alert["verdict"] == "true_positive" else 0

            X_list.append(feat.flatten())
            y_list.append(label)

        if not X_list:
            return None, None
        return np.array(X_list), np.array(y_list)

    # ------------------------------------------------------------------
    def train(self, store: FeedbackStore) -> dict | None:
        """
        Train (or retrain) the meta-classifier on all labelled data.

        Returns a metrics dict, or None if not enough data.
        """
        if not _ML_AVAILABLE:
            print("[WARNING] scikit-learn not available — cannot train meta-classifier")
            return None

        labelled = store.get_all_labelled()
        if len(labelled) < self.MIN_SAMPLES:
            return None

        X, y = self._prepare_dataset(labelled)
        if X is None or len(X) < self.MIN_SAMPLES:
            return None

        # Check we have both classes
        if len(np.unique(y)) < 2:
            print("[WARNING] Need both true_positive and false_positive samples to train")
            return None

        # Train GradientBoosting
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42,
        )

        # Cross-validation if we have enough data
        if len(X) >= 10:
            cv_folds = min(5, len(X) // 5) if len(X) >= 25 else 3
            cv_scores = cross_val_score(self.model, X, y, cv=cv_folds, scoring="f1")
            cv_f1 = float(np.mean(cv_scores))
        else:
            cv_f1 = 0.0

        # Fit on all data
        self.model.fit(X, y)
        y_pred = self.model.predict(X)

        # Compute metrics
        metrics = {
            "n_samples": len(X),
            "accuracy": round(float(accuracy_score(y, y_pred)), 3),
            "precision": round(float(precision_score(y, y_pred, zero_division=0)), 3),
            "recall": round(float(recall_score(y, y_pred, zero_division=0)), 3),
            "f1": round(float(f1_score(y, y_pred, zero_division=0)), 3),
            "cv_f1": round(cv_f1, 3),
            "feature_importances": {
                col: round(float(imp), 4)
                for col, imp in zip(FEATURE_COLS, self.model.feature_importances_)
            },
        }

        self._last_trained_count = store.count_labelled()
        self._save_model()
        store.record_training(metrics)

        print(f"[INFO] Meta-classifier trained: n={metrics['n_samples']}, "
              f"F1={metrics['f1']}, CV-F1={metrics['cv_f1']}")
        return metrics

    # ------------------------------------------------------------------
    def predict(self, signals: dict) -> dict:
        """
        Predict whether an alert is a true threat or false alarm.

        Returns:
            is_threat   : bool   — True if likely genuine
            confidence  : float  — probability of being a true threat (0–1)
        """
        if self.model is None:
            return {"is_threat": True, "confidence": 1.0}

        features = self.extract_features(signals)
        try:
            proba = self.model.predict_proba(features)[0]
            # proba[1] = probability of true_positive
            tp_confidence = float(proba[1]) if len(proba) > 1 else 1.0
        except Exception:
            return {"is_threat": True, "confidence": 1.0}

        return {
            "is_threat": tp_confidence >= self.SUPPRESSION_THRESHOLD,
            "confidence": round(tp_confidence, 3),
        }

    def get_feature_importance(self) -> dict:
        """Return feature importance from the trained model."""
        if self.model is None:
            return {}
        return {
            col: round(float(imp), 4)
            for col, imp in zip(FEATURE_COLS, self.model.feature_importances_)
        }

    def get_status(self) -> dict:
        """Return the model's current status for display."""
        return {
            "active": self.is_active(),
            "min_samples": self.MIN_SAMPLES,
            "trained_count": self._last_trained_count,
            "retrain_interval": self.RETRAIN_INTERVAL,
            "suppression_threshold": self.SUPPRESSION_THRESHOLD,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 3. ParameterOptimizer — Grid-search calibrator
# ══════════════════════════════════════════════════════════════════════════════

class ParameterOptimizer:
    """
    Grid-search optimizer that finds the best YOLO confidence, motion
    sensitivity, and fight threshold by replaying labelled feedback data.
    """

    # Search grids
    YOLO_CONF_RANGE = np.arange(0.20, 0.85, 0.05)
    MOTION_SENS_RANGE = np.arange(0.10, 0.95, 0.05)
    FIGHT_THRESH_RANGE = np.arange(0.30, 0.85, 0.05)

    def __init__(self, store: FeedbackStore):
        self.store = store

    def optimize(self) -> dict | None:
        """
        Find optimal parameter values by replaying labelled alerts.

        For each combination of (yolo_conf, fight_threshold):
        - Simulate whether each labelled alert would have been triggered
        - Compare to operator verdict
        - Pick the combination with the best F1 score

        Returns dict with optimal parameters, or None if insufficient data.
        """
        labelled = self.store.get_all_labelled()
        if len(labelled) < 15:
            return None

        # Parse signals
        records = []
        for alert in labelled:
            signals = alert.get("signals", {})
            if not signals:
                continue
            records.append({
                "yolo_confidence": signals.get("yolo_confidence", 0.5),
                "fight_score": signals.get("fight_score", 0.0),
                "motion_score": signals.get("motion_score", 0.0),
                "weapon_detected": bool(signals.get("weapon_detected", False)),
                "people_count": signals.get("people_count", 0),
                "arm_speed": signals.get("arm_speed", 0.0),
                "leg_speed": signals.get("leg_speed", 0.0),
                "proximity": signals.get("proximity", 0.0),
                "is_true": alert["verdict"] == "true_positive",
            })

        if len(records) < 10:
            return None

        best_f1 = -1.0
        best_params = None

        # Grid search over yolo_conf and fight_threshold
        # (motion_sensitivity affects motion_score indirectly, so we optimise
        #  fight_threshold which directly uses motion_score)
        for yolo_c in self.YOLO_CONF_RANGE:
            for fight_t in self.FIGHT_THRESH_RANGE:
                y_true = []
                y_pred = []

                for rec in records:
                    # Simulate: would this alert have triggered with these params?
                    would_trigger = False

                    # Weapon alert: yolo confidence must exceed threshold
                    if rec["weapon_detected"] and rec["yolo_confidence"] >= yolo_c:
                        would_trigger = True

                    # Fight alert: fight_score must exceed threshold
                    if rec["fight_score"] >= fight_t and rec["people_count"] >= 2:
                        would_trigger = True

                    y_true.append(1 if rec["is_true"] else 0)
                    y_pred.append(1 if would_trigger else 0)

                y_true = np.array(y_true)
                y_pred = np.array(y_pred)

                if np.sum(y_pred) == 0:
                    continue

                f1 = float(f1_score(y_true, y_pred, zero_division=0))
                if f1 > best_f1:
                    best_f1 = f1
                    prec = float(precision_score(y_true, y_pred, zero_division=0))
                    rec_score = float(recall_score(y_true, y_pred, zero_division=0))
                    best_params = {
                        "yolo_conf": round(float(yolo_c), 2),
                        "fight_threshold": round(float(fight_t), 2),
                        "f1_score": round(best_f1, 3),
                        "precision": round(prec, 3),
                        "recall": round(rec_score, 3),
                    }

        if best_params is None:
            return None

        # Find optimal motion sensitivity separately
        # Higher sensitivity = lower normalise_factor = more motion detected
        best_motion_f1 = -1.0
        best_motion_sens = 0.50

        for mot_s in self.MOTION_SENS_RANGE:
            # Simulate motion threshold: motion_score is normalised,
            # sensitivity affects what counts as "significant motion"
            motion_thresh = 1.0 - mot_s  # higher sensitivity = lower threshold
            y_true = []
            y_pred = []

            for rec in records:
                has_motion = rec["motion_score"] >= motion_thresh * 0.5
                # Combined: motion contributes to fight detection
                would_detect = (
                    (rec["weapon_detected"] and rec["yolo_confidence"] >= best_params["yolo_conf"])
                    or (rec["fight_score"] >= best_params["fight_threshold"]
                        and rec["people_count"] >= 2
                        and has_motion)
                )
                y_true.append(1 if rec["is_true"] else 0)
                y_pred.append(1 if would_detect else 0)

            y_true = np.array(y_true)
            y_pred = np.array(y_pred)

            if np.sum(y_pred) == 0:
                continue

            f1 = float(f1_score(y_true, y_pred, zero_division=0))
            if f1 > best_motion_f1:
                best_motion_f1 = f1
                best_motion_sens = round(float(mot_s), 2)

        best_params["motion_sensitivity"] = best_motion_sens

        # Record calibration
        self.store.record_calibration(best_params)

        print(f"[INFO] Optimal params: YOLO={best_params['yolo_conf']}, "
              f"Fight={best_params['fight_threshold']}, "
              f"Motion={best_params['motion_sensitivity']}, "
              f"F1={best_params['f1_score']}")

        return best_params

    def get_calibration_history(self, limit: int = 10) -> list[dict]:
        """Return past calibration results."""
        conn = self.store._get_conn()
        rows = conn.execute(
            "SELECT * FROM calibration_history ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: generate a unique alert ID
# ──────────────────────────────────────────────────────────────────────────────
def generate_alert_id() -> str:
    """Generate a unique alert ID."""
    return str(uuid.uuid4())[:12]
