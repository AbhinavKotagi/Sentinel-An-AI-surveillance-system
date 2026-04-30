# 🛡️ SENTINEL — Real-Time AI Threat Detection System

A complete, end-to-end surveillance dashboard powered by Streamlit, featuring live YOLO object detection, MediaPipe pose estimation for fight analysis, and a human-in-the-loop ML feedback engine to dynamically suppress false alarms.

---

## 🔥 Key Features

* **Real-Time AI Pipeline**: Integrates YOLOv8 for person/weapon detection and MediaPipe for skeletal pose estimation.
* **Fight & Motion Analysis**: Calculates arm/leg speed and body proximity to accurately detect altercations.
* **Human-in-the-Loop Feedback**: A dedicated Review UI to mark alerts as True/False Positives.
* **Auto-Calibration & ML Filtering**: Trains a `GradientBoostingClassifier` on the fly to suppress likely false alarms and provides one-click parameter optimization for the dashboard sliders.
* **Local SQLite Logging**: All alerts, snapshots, and raw detection signals are persisted to a local database.

---

## 🚀 Quick Start

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first time you run the app, Ultralytics will automatically download the YOLOv8 model weights.

---

### 2 — Run the app

```bash
streamlit run app.py
```

The browser will open automatically at **http://localhost:8501**.

---

## 🖥️ Usage Guide

### Normal Monitoring Mode
1. Open the sidebar and choose **Webcam** or **Video File**.
2. (Optional) Upload an `.mp4 / .avi / .mov` file.
3. Adjust **YOLO Confidence** and **Motion Sensitivity**.
4. Click **▶ START MONITORING**.
5. Watch the threat levels, bounding boxes, skeletal overlays, and alerts update in real-time.

### ML Feedback & Review Mode
1. Click **📋 REVIEW ALERTS** in the sidebar.
2. The main feed will switch to an Accuracy Dashboard showing precision, recall, and feature importances.
3. Review pending alerts. Click **✅ TRUE THREAT** or **❌ FALSE ALARM** on the generated snapshots to log feedback.
4. Once you have labelled at least 30 samples, the system will automatically train the ML filter and suppress false alarms.
5. Click **🎯 AUTO-CALIBRATE** to replay your labelled data and automatically find the optimal slider thresholds for maximum accuracy.

---

## ⚡ Threat Logic

| Condition | Level |
|-----------|-------|
| Weapon + Multiple People | 🔴 CRITICAL |
| Weapon detected only | 🟠 HIGH |
| Active Fight Detected | 🟠 HIGH |
| ≥2 people + High Motion | 🟡 MEDIUM |
| High motion only | 🟡 MEDIUM |
| None of the above | 🟢 SAFE |

---

## 🏗️ Architecture

```
backend/
├── detector.py      — YOLOv8 inference (Persons, Weapons)
├── pose.py          — MediaPipe skeletal tracking
├── motion.py        — Structural Similarity Index (SSIM) based motion detection
├── logic.py         — Fight detection math (speed/proximity) & threat aggregation
├── alert.py         — Alert generation, JSON snapshot saving
├── feedback.py      — SQLite DB, GradientBoosting classifier, Parameter GridSearch
└── review_ui.py     — Streamlit feedback dashboard components
app.py               — Main Streamlit app layout & real-time video loop
```

---

## ⚙️ How the ML Feedback Engine Works

Instead of retraining the heavy computer vision models (which takes hours), Sentinel uses a lightweight meta-classifier approach:

1. **Signal Extraction**: During a threat, 10 key signals (YOLO confidence, arm speed, motion score, time of day, etc.) are extracted.
2. **Labeling**: An operator reviews the alert snapshot and labels it.
3. **Training**: A `GradientBoostingClassifier` trains on this tabular data in milliseconds.
4. **Inference**: On future frames, if the rule-engine flags a threat, the meta-classifier evaluates the signals. If it determines the threat confidence is < 40%, the alert is silently suppressed.
