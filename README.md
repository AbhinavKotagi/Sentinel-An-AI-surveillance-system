# 🛡️ SENTINEL — Real-Time AI Threat Detection System

A hackathon-ready Streamlit surveillance dashboard with simulated AI detections,
real-time threat levels, bounding-box overlays, and a live alert log.

---

## Quick Start

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** On headless servers (no display) use `opencv-python-headless`.  
> If you have a local desktop with a webcam you can also use `opencv-python`.

---

### 2 — Run the app

```bash
streamlit run app.py
```

The browser will open automatically at **http://localhost:8501**.

---

## Usage

| Step | Action |
|------|--------|
| 1 | Open the sidebar and choose **Webcam** or **Video File** |
| 2 | (Optional) Upload an `.mp4 / .avi / .mov` file |
| 3 | Adjust **Weapon Sensitivity**, **Motion Threshold**, and **Target FPS** |
| 4 | Click **▶ START MONITORING** |
| 5 | Watch threat levels, bounding boxes, and alerts update in real time |
| 6 | Click **■ STOP MONITORING** to pause |
| 7 | Click **🗑 CLEAR ALERT LOG** to reset history |

---

## Threat Logic

| Condition | Level |
|-----------|-------|
| Weapon + multiple people | 🔴 CRITICAL |
| Weapon detected only | 🟠 HIGH |
| ≥2 people + high motion | 🟡 MEDIUM |
| High motion only | 🟡 MEDIUM |
| None of the above | 🟢 SAFE |

---

## Architecture

```
app.py
├── simulate_detection()   — Frame-diff motion + randomised detections
├── compute_threat()       — Threat level logic
├── draw_overlays()        — OpenCV bounding boxes + HUD elements
└── Streamlit layout       — Sidebar · Video · Threat panel · Alert log
```

---

## Tips for Demo Day

* Run on a **laptop with a webcam** for maximum impact — wave your hands to
  trigger high motion scores.
* Use a pre-recorded video of a busy scene to guarantee detections every run.
* Lower **Target FPS** to `8–10` if the machine is slow; the UI stays smooth.
* The **Weapon Sensitivity** slider lets you force frequent HIGH/CRITICAL alerts
  for a dramatic live demo — crank it up just before presenting.

---

## Customisation

Replace `simulate_detection()` with calls to your real ML backend
(YOLO, OpenCV DNN, etc.) — the rest of the pipeline stays unchanged.

```python
# Example: drop-in real backend
def simulate_detection(frame, prev_gray):
    results = your_model.predict(frame)
    return {
        "num_people":      results.person_count,
        "weapon_detected": results.weapon_found,
        "motion_score":    compute_motion(frame, prev_gray),
        "detections":      results.boxes,
        "gray":            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
    }
```
