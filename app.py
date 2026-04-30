import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

import streamlit as st
import cv2
import numpy as np
import time
from datetime import datetime
from collections import deque

from detector import ThreatDetector
from pose import PoseEstimator
from motion import MotionDetector
from logic import FightDetector, compute_threat_level, classify_detection_type
from alert import AlertSystem
from feedback import FeedbackStore, ThreatMetaClassifier, ParameterOptimizer
from review_ui import render_review_panel, render_ml_status_sidebar, render_calibration_result

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SENTINEL · AI Threat Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700;900&display=swap');
  html, body, [data-testid="stAppViewContainer"] {
    background: #050a0f !important; color: #c8d8e8 !important;
    font-family: 'Exo 2', sans-serif !important;
  }
  [data-testid="stSidebar"] {
    background: #070d14 !important; border-right: 1px solid #0d2035 !important;
  }
  [data-testid="stSidebar"] * { color: #8aaac0 !important; }
  [data-testid="stAppViewContainer"]::before {
    content: ""; position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(0deg, transparent, transparent 2px,
      rgba(0,255,180,0.012) 2px, rgba(0,255,180,0.012) 4px);
    pointer-events: none; z-index: 9999;
  }
  .sentinel-header {
    display: flex; align-items: center; gap: 16px;
    padding: 18px 0 10px; border-bottom: 1px solid #0d2035; margin-bottom: 20px;
  }
  .sentinel-logo {
    font-family: 'Share Tech Mono', monospace; font-size: 26px;
    color: #00ffa3; letter-spacing: 4px; text-shadow: 0 0 20px rgba(0,255,163,0.5);
  }
  .sentinel-sub {
    font-size: 11px; color: #3a6080; letter-spacing: 3px;
    text-transform: uppercase; margin-top: 2px;
  }
  .live-dot {
    width: 10px; height: 10px; border-radius: 50%; background: #00ffa3;
    box-shadow: 0 0 12px #00ffa3; animation: pulse 1.2s ease-in-out infinite;
    display: inline-block; margin-right: 6px;
  }
  @keyframes pulse {
    0%,100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.7); }
  }
  .threat-badge {
    border-radius: 6px; padding: 20px 28px; text-align: center;
    font-family: 'Share Tech Mono', monospace; font-size: 32px;
    letter-spacing: 6px; font-weight: 700; margin-bottom: 14px;
    border: 1px solid; position: relative; overflow: hidden;
  }
  .threat-badge::after {
    content: ""; position: absolute; inset: 0; background: currentColor; opacity: 0.07;
  }
  .threat-SAFE { color:#00ffa3; border-color:#00ffa340; text-shadow:0 0 18px #00ffa380; }
  .threat-MEDIUM { color:#ffe066; border-color:#ffe06640; text-shadow:0 0 18px #ffe06680; }
  .threat-HIGH { color:#ff8c00; border-color:#ff8c0040; text-shadow:0 0 18px #ff8c0080; }
  .threat-CRITICAL { color:#ff2255; border-color:#ff225540; text-shadow:0 0 18px #ff225580;
    animation: critblink 0.7s step-end infinite; }
  @keyframes critblink { 0%,100%{opacity:1} 50%{opacity:.6} }
  .threat-label {
    font-size: 11px; letter-spacing: 3px; color: #3a6080; margin-bottom: 6px;
    font-family: 'Share Tech Mono', monospace;
  }
  .stat-card {
    background: #080f18; border: 1px solid #0d2035;
    border-radius: 8px; padding: 14px 18px; margin-bottom: 10px;
  }
  .stat-label {
    font-size: 10px; letter-spacing: 2px; color: #3a6080;
    text-transform: uppercase; font-family: 'Share Tech Mono', monospace;
  }
  .stat-value {
    font-size: 26px; font-weight: 700; color: #c8d8e8;
    font-family: 'Share Tech Mono', monospace; line-height: 1.1;
  }
  .stat-value.accent { color: #00ffa3; }
  .alert-item {
    border-left: 3px solid; padding: 10px 14px; margin-bottom: 8px;
    background: #080f18; border-radius: 0 6px 6px 0; font-size: 13px;
  }
  .alert-MEDIUM { border-color: #ffe066; }
  .alert-HIGH { border-color: #ff8c00; }
  .alert-CRITICAL { border-color: #ff2255; }
  .alert-time { font-family:'Share Tech Mono',monospace; font-size:11px; color:#3a6080; }
  .alert-level { font-weight:700; font-size:12px; letter-spacing:2px; }
  .alert-MEDIUM .alert-level { color:#ffe066; }
  .alert-HIGH .alert-level { color:#ff8c00; }
  .alert-CRITICAL .alert-level { color:#ff2255; }
  .alert-msg { color:#8aaac0; margin-top:2px; font-size:12px; }
  .section-title {
    font-family: 'Share Tech Mono', monospace; font-size: 11px;
    letter-spacing: 3px; color: #3a6080; text-transform: uppercase;
    margin: 16px 0 10px; display: flex; align-items: center; gap: 8px;
  }
  .section-title::after { content: ""; flex: 1; height: 1px; background: #0d2035; }
  .motion-bar-bg {
    background: #0d2035; border-radius: 4px; height: 6px;
    margin-top: 6px; overflow: hidden;
  }
  .motion-bar-fill { height: 100%; border-radius: 4px; transition: width 0.3s ease; }
  .signal-bar {
    background: #080f18; border: 1px solid #0d2035; border-radius: 6px;
    padding: 8px 12px; margin-bottom: 6px;
  }
  .signal-label {
    font-size: 9px; letter-spacing: 2px; color: #3a6080;
    font-family: 'Share Tech Mono', monospace; text-transform: uppercase;
  }
  .signal-val {
    font-size: 16px; font-weight: 600; color: #c8d8e8;
    font-family: 'Share Tech Mono', monospace;
  }
  .json-box {
    background: #060c14; border: 1px solid #0d2035; border-radius: 6px;
    padding: 10px 14px; font-family: 'Share Tech Mono', monospace;
    font-size: 11px; color: #4a8aaa; line-height: 1.6; white-space: pre;
  }
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: #050a0f; }
  ::-webkit-scrollbar-thumb { background: #0d2035; border-radius: 2px; }
  .sys-clock {
    font-family: 'Share Tech Mono', monospace; font-size: 12px;
    color: #3a6080; letter-spacing: 2px;
  }
  [data-testid="stImage"] img { border-radius: 6px !important; }
  [data-testid="stButton"] > button {
    background: #0a1a28 !important; color: #00ffa3 !important;
    border: 1px solid #00ffa340 !important; border-radius: 6px !important;
    font-family: 'Share Tech Mono', monospace !important;
    letter-spacing: 2px !important; width: 100%;
    font-size: 13px !important; padding: 10px !important;
    transition: all 0.2s !important;
  }
  [data-testid="stButton"] > button:hover {
    background: #00ffa315 !important; border-color: #00ffa3 !important;
    box-shadow: 0 0 14px #00ffa330 !important;
  }
  [data-testid="stSelectbox"] label,
  [data-testid="stRadio"] label { color: #3a6080 !important; font-size:11px; letter-spacing:2px; }
  hr { border-color: #0d2035 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ──────────────────────────────────────────────────────
defaults = {
    "running": False, "alert_history": deque(maxlen=50),
    "frame_count": 0, "total_alerts": 0, "source_type": "Webcam",
    "review_mode": False, "auto_suppress": True,
    "fight_threshold": None,  # None = use default
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Cache backend modules ──────────────────────────────────────────────────
@st.cache_resource
def load_detector(conf):
    return ThreatDetector(conf_threshold=conf)

@st.cache_resource
def load_pose():
    return PoseEstimator()

@st.cache_resource
def load_motion():
    return MotionDetector()

@st.cache_resource
def load_fight():
    return FightDetector()

@st.cache_resource
def load_alert():
    return AlertSystem()

@st.cache_resource
def load_feedback_store():
    return FeedbackStore()

@st.cache_resource
def load_meta_classifier():
    return ThreatMetaClassifier()

# ─── Draw overlays ───────────────────────────────────────────────────────────
def draw_overlays(frame, detections, threat_level):
    out = frame.copy()
    h, w = out.shape[:2]
    tc_map = {
        "SAFE": (0,255,163), "MEDIUM": (255,224,80),
        "HIGH": (255,140,0), "CRITICAL": (255,34,85),
    }
    tc = tc_map.get(threat_level, (200,200,200))

    for obj in detections:
        x1, y1, x2, y2 = obj["box"]
        c = obj["color"]
        cv2.rectangle(out, (x1,y1), (x2,y2), c, 2)
        cs = 12
        for cx,cy,dx,dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(out, (cx,cy), (cx+dx*cs,cy), c, 3)
            cv2.line(out, (cx,cy), (cx,cy+dy*cs), c, 3)
        label = f"{obj['label']}  {obj['confidence']:.0%}"
        (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1,y1-th-10), (x1+tw+10,y1), c, -1)
        cv2.putText(out, label, (x1+5,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (5,10,15), 1, cv2.LINE_AA)

    tl_text = threat_level
    (ttw,tth),_ = cv2.getTextSize(tl_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    tx = w - ttw - 18
    cv2.rectangle(out, (tx-8,8), (w-8,tth+18), tc, -1)
    cv2.putText(out, tl_text, (tx,tth+12), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (5,10,15), 2, cv2.LINE_AA)

    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(out, ts, (10,h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60,100,130), 1, cv2.LINE_AA)

    if threat_level == "CRITICAL":
        cv2.rectangle(out, (0,0), (w-1,h-1), (255,34,85), 5)

    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 20px;">
      <div style="font-family:'Share Tech Mono',monospace; font-size:20px;
                  color:#00ffa3; letter-spacing:4px; text-shadow:0 0 16px #00ffa360;">
        🛡️ SENTINEL
      </div>
      <div style="font-size:9px; color:#3a6080; letter-spacing:3px; margin-top:4px;">
        AI SURVEILLANCE SYSTEM v3.0
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">VIDEO SOURCE</div>', unsafe_allow_html=True)
    source = st.radio("", ["Webcam", "Video File"], label_visibility="collapsed", key="source_radio")
    uploaded_file = None
    if source == "Video File":
        uploaded_file = st.file_uploader("Upload video", type=["mp4","avi","mov","mkv"],
                                         label_visibility="collapsed")

    st.markdown('<div class="section-title">AI ENGINE</div>', unsafe_allow_html=True)
    yolo_conf = st.slider("YOLO Confidence", 0.10, 0.90, 0.40, 0.05, format="%.2f")
    motion_sensitivity = st.slider("Motion Sensitivity", 0.0, 1.0, 0.50, 0.05, format="%.2f",
                                    help="0 = least sensitive, 1 = most sensitive")
    target_fps = st.slider("Target FPS", 1, 30, 12, 1)

    st.markdown('<div class="section-title">CONTROLS</div>', unsafe_allow_html=True)
    if not st.session_state.running:
        if st.button("▶  START MONITORING"):
            st.session_state.running = True
            st.session_state.frame_count = 0
            st.rerun()
    else:
        if st.button("■  STOP MONITORING"):
            st.session_state.running = False
            st.rerun()
    if st.button("🗑  CLEAR ALERT LOG"):
        st.session_state.alert_history.clear()
        st.session_state.total_alerts = 0
        st.rerun()

    st.markdown('<div class="section-title">ML FEEDBACK</div>', unsafe_allow_html=True)

    feedback_store = load_feedback_store()
    meta_clf = load_meta_classifier()

    if st.button("📋  REVIEW ALERTS"):
        st.session_state.review_mode = not st.session_state.review_mode
        st.rerun()

    st.session_state.auto_suppress = st.checkbox(
        "🤖 Auto-suppress false alarms",
        value=st.session_state.get("auto_suppress", True),
        help="When ML model is active, automatically suppress likely false alarms",
    )

    if st.button("🎯  AUTO-CALIBRATE"):
        optimizer = ParameterOptimizer(feedback_store)
        result = optimizer.optimize()
        if result:
            st.session_state["calibrated"] = result
            st.rerun()
        else:
            st.warning("⚠ Need 15+ labelled alerts")

    if st.button("🧠  TRAIN MODEL"):
        if feedback_store.count_labelled() >= meta_clf.MIN_SAMPLES:
            metrics = meta_clf.train(feedback_store)
            if metrics:
                st.success(f"F1={metrics['f1']}")
        else:
            st.warning(f"⚠ Need {meta_clf.MIN_SAMPLES}+ labelled alerts")

    if st.button("🗑  CLEAR FEEDBACK DB"):
        feedback_store.clear_all()
        meta_clf.clear_model()
        st.session_state.pop("calibrated", None)
        st.rerun()

    render_ml_status_sidebar(meta_clf, feedback_store)

    # Show calibration result if available
    if "calibrated" in st.session_state:
        render_calibration_result(st.session_state["calibrated"])
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("✅ APPLY"):
                r = st.session_state["calibrated"]
                st.session_state["applied_yolo"] = r["yolo_conf"]
                st.session_state["applied_motion"] = r["motion_sensitivity"]
                st.session_state["fight_threshold"] = r["fight_threshold"]
                del st.session_state["calibrated"]
                st.rerun()
        with col_b:
            if st.button("✗ DISMISS"):
                del st.session_state["calibrated"]
                st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="font-size:10px; color:#1e3a50; text-align:center; line-height:1.8;">
      SENTINEL © 2025<br>Real-Time Threat Detection<br>
      <span style="color:#00ffa340;">● AI ENGINE ACTIVE</span>
    </div>
    """, unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────
live_html = ('<span class="live-dot"></span>'
             '<span style="color:#00ffa3;font-family:Share Tech Mono,monospace;'
             'font-size:12px;letter-spacing:2px;">LIVE</span>')
standby_html = ('<span style="color:#1e3a50;font-family:Share Tech Mono,monospace;'
                'font-size:12px;letter-spacing:2px;">■ STANDBY</span>')
st.markdown(f"""
<div class="sentinel-header">
  <div>
    <div class="sentinel-logo">🛡 SENTINEL</div>
    <div class="sentinel-sub">Real-Time AI Threat Detection System</div>
  </div>
  <div style="margin-left:auto; text-align:right;">
    {live_html if st.session_state.running else standby_html}
    <div class="sys-clock">{datetime.now().strftime("%Y-%m-%d  %H:%M:%S")}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Main Layout ─────────────────────────────────────────────────────────────
if st.session_state.get("review_mode", False):
    # ── REVIEW MODE ──
    feedback_store = load_feedback_store()
    meta_clf = load_meta_classifier()
    render_review_panel(feedback_store, meta_clf)
else:
    # ── NORMAL MODE ──
    col_video, col_panel = st.columns([3, 2], gap="large")

    with col_video:
        st.markdown('<div class="section-title">CAMERA FEED</div>', unsafe_allow_html=True)
        video_placeholder = st.empty()
        if not st.session_state.running:
            standby = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(standby, "SYSTEM STANDBY", (140,170), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (13,32,48), 2)
            cv2.putText(standby, "Press START MONITORING to begin", (100,210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30,80,100), 1)
            for i in range(0, 640, 40):
                cv2.line(standby, (i,0), (i,360), (8,20,32), 1)
            for i in range(0, 360, 40):
                cv2.line(standby, (0,i), (640,i), (8,20,32), 1)
            video_placeholder.image(standby, channels="BGR", use_column_width=True)

        st.markdown('<div class="section-title">FRAME DATA · JSON</div>', unsafe_allow_html=True)
        json_placeholder = st.empty()

    with col_panel:
        st.markdown('<div class="section-title">THREAT STATUS</div>', unsafe_allow_html=True)
        threat_placeholder = st.empty()
        stats_placeholder = st.empty()
        st.markdown('<div class="section-title">FIGHT ANALYSIS</div>', unsafe_allow_html=True)
        fight_placeholder = st.empty()
        st.markdown('<div class="section-title">ACTIVE ALERTS</div>', unsafe_allow_html=True)
        active_placeholder = st.empty()
        st.markdown('<div class="section-title">ALERT HISTORY</div>', unsafe_allow_html=True)
        history_placeholder = st.empty()

# ─── Render Panel ────────────────────────────────────────────────────────────
def render_panel(threat_level="SAFE", frame_data=None, fight_data=None, reason="", active_msgs=None):
    motion = frame_data["motion_score"] if frame_data else 0.0
    people = frame_data["people"] if frame_data else 0
    weapon = "YES" if (frame_data and frame_data["weapon"]) else "NO"
    wc = "#ff2255" if weapon == "YES" else "#00ffa3"
    mbar = int(motion * 100)
    mbar_c = "#ff2255" if motion > 0.65 else ("#ffe066" if motion > 0.35 else "#00ffa3")

    threat_placeholder.markdown(f"""
    <div class="threat-label">CURRENT THREAT LEVEL</div>
    <div class="threat-badge threat-{threat_level}">{threat_level}</div>
    """, unsafe_allow_html=True)

    stats_placeholder.markdown(f"""
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
      <div class="stat-card">
        <div class="stat-label">PERSONS</div>
        <div class="stat-value accent">{people}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">WEAPON</div>
        <div class="stat-value" style="color:{wc};">{weapon}</div>
      </div>
      <div class="stat-card" style="grid-column:1/-1;">
        <div class="stat-label">MOTION SCORE</div>
        <div class="stat-value">{motion:.0%}</div>
        <div class="motion-bar-bg">
          <div class="motion-bar-fill" style="width:{mbar}%; background:{mbar_c};"></div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Fight analysis signals
    if fight_data:
        arm_s = fight_data.get("arm_speed", 0)
        leg_s = fight_data.get("leg_speed", 0)
        prox = fight_data.get("proximity", 0)
        fscore = fight_data.get("fight_score", 0)
        fdet = fight_data.get("fight_detected", False)
        pclose = fight_data.get("people_close", False)
        fc = "#ff2255" if fdet else "#00ffa3"
        arm_bar = int(arm_s * 100)
        leg_bar = int(leg_s * 100)
        prox_bar = int(prox * 100)
        fs_bar = int(fscore * 100)
        pc_color = "#ff8c00" if pclose else "#3a6080"
        fight_placeholder.markdown(f"""
        <div class="signal-bar">
          <div class="signal-label">ARM SPEED</div>
          <div class="signal-val">{arm_s:.3f}</div>
          <div class="motion-bar-bg"><div class="motion-bar-fill" style="width:{arm_bar}%;background:#4a9eff;"></div></div>
        </div>
        <div class="signal-bar">
          <div class="signal-label">LEG SPEED</div>
          <div class="signal-val">{leg_s:.3f}</div>
          <div class="motion-bar-bg"><div class="motion-bar-fill" style="width:{leg_bar}%;background:#aa66ff;"></div></div>
        </div>
        <div class="signal-bar">
          <div class="signal-label">PROXIMITY</div>
          <div class="signal-val">{prox:.3f}</div>
          <div class="motion-bar-bg"><div class="motion-bar-fill" style="width:{prox_bar}%;background:#ff8c00;"></div></div>
        </div>
        <div class="signal-bar">
          <div class="signal-label">PEOPLE CLOSE</div>
          <div class="signal-val" style="color:{pc_color};font-size:16px;">{'YES' if pclose else 'NO'}</div>
        </div>
        <div class="signal-bar">
          <div class="signal-label">FIGHT SCORE</div>
          <div class="signal-val" style="color:{fc};">{fscore:.3f}</div>
          <div class="motion-bar-bg"><div class="motion-bar-fill" style="width:{fs_bar}%;background:{fc};"></div></div>
        </div>
        <div class="signal-bar" style="text-align:center;">
          <div class="signal-label">FIGHT DETECTED</div>
          <div class="signal-val" style="color:{fc};font-size:20px;">{'⚠ YES' if fdet else '● NO'}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        fight_placeholder.markdown(
            '<div style="color:#1e3a50;font-size:12px;font-family:Share Tech Mono,monospace;'
            'letter-spacing:2px;padding:10px 0;">● AWAITING DATA</div>', unsafe_allow_html=True)

    if active_msgs:
        msgs_html = "".join(
            f'<div class="alert-item alert-{threat_level}">'
            f'<span class="alert-level">{threat_level}</span>'
            f'<div class="alert-msg">⚠ {m}</div></div>'
            for m in active_msgs
        )
        active_placeholder.markdown(msgs_html, unsafe_allow_html=True)
    else:
        active_placeholder.markdown(
            '<div style="color:#1e3a50;font-size:12px;font-family:Share Tech Mono,monospace;'
            'letter-spacing:2px;padding:10px 0;">● NO ACTIVE ALERTS</div>', unsafe_allow_html=True)

    history = list(st.session_state.alert_history)[::-1]
    if history:
        hist_html = "".join(
            f'<div class="alert-item alert-{a["level"]}">'
            f'<div class="alert-time">{a["time"]}</div>'
            f'<span class="alert-level">{a["level"]}</span>'
            f'<div class="alert-msg">{a["msg"]}</div></div>'
            for a in history[:12]
        )
        history_placeholder.markdown(hist_html, unsafe_allow_html=True)
    else:
        history_placeholder.markdown(
            '<div style="color:#1e3a50;font-size:12px;font-family:Share Tech Mono,monospace;'
            'letter-spacing:2px;padding:10px 0;">NO ALERTS RECORDED</div>', unsafe_allow_html=True)

if not st.session_state.get("review_mode", False):
    render_panel()

# ─── Main Video Loop ─────────────────────────────────────────────────────────
if st.session_state.running and not st.session_state.get("review_mode", False):
    detector = load_detector(yolo_conf)
    pose_est = load_pose()
    motion_det = load_motion()
    fight_det = load_fight()
    alert_sys = load_alert()
    feedback_store = load_feedback_store()
    meta_clf = load_meta_classifier()

    # Use calibrated thresholds if applied
    effective_yolo = st.session_state.get("applied_yolo", yolo_conf)
    effective_motion = st.session_state.get("applied_motion", motion_sensitivity)
    effective_fight_thresh = st.session_state.get("fight_threshold", None)

    cap = None
    tmp_path = None

    try:
        if source == "Webcam":
            cap = cv2.VideoCapture(0)
        else:
            if uploaded_file is not None:
                import tempfile
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp.write(uploaded_file.read())
                tmp.flush()
                tmp_path = tmp.name
                tmp.close()
                cap = cv2.VideoCapture(tmp_path)
            else:
                st.warning("⚠ Please upload a video file first.")
                st.session_state.running = False
                st.stop()

        if not cap.isOpened():
            st.error("Cannot open video source.")
            st.session_state.running = False
            st.stop()

        frame_delay = 1.0 / target_fps

        while st.session_state.running:
            t0 = time.time()
            ret, frame = cap.read()

            if not ret:
                if source == "Video File":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            frame = cv2.resize(frame, (640, 360))

            # ── Real AI Pipeline ──
            det_result = detector.detect(frame)
            detections = det_result["detections"]
            people_count = det_result["people_count"]
            weapon_detected = det_result["weapon_detected"]

            pose_result = pose_est.process(frame)
            landmarks = pose_result["landmarks"]
            frame = pose_result["frame"]

            motion_det.set_sensitivity(effective_motion)
            motion_score = motion_det.compute(frame)

            fight_result = fight_det.detect_fight(
                landmarks, detections, motion_score,
                threshold_override=effective_fight_thresh,
            )
            fight_detected = fight_result["fight_detected"]

            threat_level, reason = compute_threat_level(weapon_detected, fight_detected, people_count)
            det_type = classify_detection_type(weapon_detected, fight_detected)

            # ── Build ML signal vector ──
            max_det_conf = max((d["confidence"] for d in detections), default=0.0)
            signals = {
                "yolo_confidence": round(max_det_conf, 3),
                "people_count": people_count,
                "weapon_detected": int(weapon_detected),
                "motion_score": round(motion_score, 3),
                "arm_speed": fight_result.get("arm_speed", 0.0),
                "leg_speed": fight_result.get("leg_speed", 0.0),
                "proximity": fight_result.get("proximity", 0.0),
                "fight_score": fight_result.get("fight_score", 0.0),
                "threat_level_enc": {"SAFE": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}.get(threat_level, 0),
                "hour_of_day": datetime.now().hour,
            }

            # ── ML Meta-Classifier: suppress likely false alarms ──
            ml_prediction = None
            if threat_level != "SAFE" and meta_clf.is_active() and st.session_state.get("auto_suppress", True):
                prediction = meta_clf.predict(signals)
                ml_prediction = prediction["confidence"]
                if not prediction["is_threat"]:
                    threat_level = "SAFE"
                    reason = ""

            # ── Frame data dict ──
            frame_data = {
                "people": people_count,
                "weapon": weapon_detected,
                "motion_score": round(motion_score, 3),
                "fight": fight_detected,
                "threat": threat_level,
            }

            # ── Alert + Feedback logging ──
            # Optionally log SAFE frames if there's activity to allow reviewing missed threats
            log_safe_frame = (threat_level == "SAFE" and (people_count > 0 or motion_score > 0.3))
            alert_record = alert_sys.trigger(
                frame, threat_level, reason,
                detection_type=det_type, signals=signals,
                log_safe=log_safe_frame
            )
            if alert_record:
                if alert_record["level"] != "SAFE":
                    st.session_state.alert_history.append({
                        "time": alert_record["timestamp"],
                        "level": alert_record["level"],
                        "msg": alert_record["reason"],
                    })
                    st.session_state.total_alerts += 1
                # Log to feedback DB for ML training
                feedback_store.record_alert(
                    alert_id=alert_record["alert_id"],
                    timestamp=alert_record["timestamp_iso"],
                    threat_level=alert_record["level"],
                    reason=alert_record["reason"],
                    snapshot_path=alert_record["snapshot"],
                    det_type=alert_record["detection_type"],
                    signals=alert_record["signals"],
                    ml_prediction=ml_prediction,
                )

                # Auto-retrain if enough new feedback
                if meta_clf.should_retrain(feedback_store):
                    meta_clf.train(feedback_store)

            # ── Draw & display ──
            rgb_frame = draw_overlays(frame, detections, threat_level)
            video_placeholder.image(rgb_frame, use_column_width=True)

            # ── JSON display ──
            json_str = (
                f'{{\n'
                f'  "people":       {frame_data["people"]},\n'
                f'  "weapon":       {str(frame_data["weapon"]).lower()},\n'
                f'  "motion_score": {frame_data["motion_score"]},\n'
                f'  "fight":        {str(frame_data["fight"]).lower()},\n'
                f'  "threat":       "{frame_data["threat"]}"\n'
                f'}}'
            )
            json_placeholder.markdown(f'<div class="json-box">{json_str}</div>', unsafe_allow_html=True)

            # ── Build alert messages for panel ──
            active_msgs = []
            if threat_level != "SAFE":
                active_msgs = [reason]

            st.session_state.frame_count += 1
            render_panel(threat_level, frame_data, fight_result, reason,
                         active_msgs if threat_level != "SAFE" else None)

            elapsed = time.time() - t0
            if elapsed < frame_delay:
                time.sleep(frame_delay - elapsed)

    finally:
        if cap:
            cap.release()
        if tmp_path:
            try: os.unlink(tmp_path)
            except: pass

