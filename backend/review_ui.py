"""
review_ui.py — Streamlit UI components for the threat review & ML feedback system.
"""
import streamlit as st
import os
from datetime import datetime

def render_review_css():
    st.markdown("""
    <style>
    .review-card{background:#080f18;border:1px solid #0d2035;border-radius:8px;padding:16px;margin-bottom:12px;}
    .review-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;}
    .review-id{font-family:'Share Tech Mono',monospace;font-size:10px;color:#3a6080;letter-spacing:2px;}
    .review-signals{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin:10px 0;}
    .review-sig{background:#060c14;border:1px solid #0d2035;border-radius:4px;padding:6px 8px;}
    .review-sig-label{font-size:8px;color:#3a6080;letter-spacing:1px;text-transform:uppercase;font-family:'Share Tech Mono',monospace;}
    .review-sig-val{font-size:14px;color:#c8d8e8;font-family:'Share Tech Mono',monospace;}
    .ml-status-box{background:#080f18;border:1px solid #0d2035;border-radius:6px;padding:10px 14px;margin:8px 0;}
    .ml-active{color:#00ffa3;} .ml-inactive{color:#ff8c00;}
    .accuracy-gauge{background:#0d2035;border-radius:4px;height:8px;margin-top:4px;overflow:hidden;}
    .accuracy-fill{height:100%;border-radius:4px;transition:width 0.3s ease;}
    .calib-row{display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #0d2035;}
    .calib-label{font-size:10px;color:#3a6080;font-family:'Share Tech Mono',monospace;letter-spacing:1px;}
    .calib-val{font-size:13px;color:#00ffa3;font-family:'Share Tech Mono',monospace;}
    .feat-bar-bg{background:#0d2035;border-radius:3px;height:6px;flex:1;margin-left:8px;}
    .feat-bar-fill{height:100%;border-radius:3px;background:#4a9eff;}
    .feat-row{display:flex;align-items:center;margin-bottom:4px;}
    .feat-name{font-size:9px;color:#3a6080;font-family:'Share Tech Mono',monospace;width:110px;letter-spacing:1px;}
    </style>
    """, unsafe_allow_html=True)


def render_ml_status_sidebar(meta_clf, feedback_store):
    """Render ML model status in sidebar."""
    status = meta_clf.get_status()
    stats = feedback_store.get_stats()
    n_labelled = stats["labelled"]
    n_pending = stats["pending"]

    if status["active"]:
        icon, cls, txt = "●", "ml-active", "MODEL ACTIVE"
    elif n_labelled >= status["min_samples"]:
        icon, cls, txt = "◐", "ml-inactive", "READY TO TRAIN"
    else:
        need = status["min_samples"] - n_labelled
        icon, cls, txt = "○", "ml-inactive", f"NEED {need} MORE LABELS"

    st.markdown(f"""
    <div class="ml-status-box">
      <div style="font-size:9px;color:#3a6080;letter-spacing:2px;font-family:'Share Tech Mono',monospace;">ML ENGINE</div>
      <div class="{cls}" style="font-size:13px;font-family:'Share Tech Mono',monospace;">{icon} {txt}</div>
      <div style="font-size:10px;color:#3a6080;margin-top:4px;">
        Labelled: {n_labelled} · Pending: {n_pending}
      </div>
    </div>
    """, unsafe_allow_html=True)

    if stats["precision"] is not None:
        pct = int(stats["precision"] * 100)
        pc = "#00ffa3" if pct >= 70 else ("#ffe066" if pct >= 50 else "#ff2255")
        st.markdown(f"""
        <div class="ml-status-box">
          <div style="font-size:9px;color:#3a6080;letter-spacing:2px;font-family:'Share Tech Mono',monospace;">PRECISION</div>
          <div style="font-size:18px;color:{pc};font-family:'Share Tech Mono',monospace;">{pct}%</div>
          <div class="accuracy-gauge"><div class="accuracy-fill" style="width:{pct}%;background:{pc};"></div></div>
          <div style="font-size:9px;color:#3a6080;margin-top:4px;">TP: {stats['true_positives']} · FP: {stats['false_positives']}</div>
        </div>
        """, unsafe_allow_html=True)


def render_review_panel(feedback_store, meta_clf):
    """Render the full review panel with pending alerts and accuracy dashboard."""
    render_review_css()
    pending = feedback_store.get_pending_reviews(limit=20)
    stats = feedback_store.get_stats()

    # ── Accuracy Dashboard ──
    st.markdown('<div class="section-title">📊 ACCURACY DASHBOARD</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class="stat-card"><div class="stat-label">TOTAL ALERTS</div>
        <div class="stat-value accent">{stats['total_alerts']}</div></div>""", unsafe_allow_html=True)
    with col2:
        tp = stats['true_positives']
        st.markdown(f"""<div class="stat-card"><div class="stat-label">TRUE POSITIVES</div>
        <div class="stat-value" style="color:#00ffa3;">{tp}</div></div>""", unsafe_allow_html=True)
    with col3:
        fp = stats['false_positives']
        st.markdown(f"""<div class="stat-card"><div class="stat-label">FALSE POSITIVES</div>
        <div class="stat-value" style="color:#ff2255;">{fp}</div></div>""", unsafe_allow_html=True)

    # Per-category breakdown
    if stats["by_category"]:
        st.markdown('<div class="section-title">PER-CATEGORY PRECISION</div>', unsafe_allow_html=True)
        for cat, data in stats["by_category"].items():
            pct = int(data["precision"] * 100)
            pc = "#00ffa3" if pct >= 70 else ("#ffe066" if pct >= 50 else "#ff2255")
            st.markdown(f"""
            <div class="ml-status-box">
              <div style="display:flex;justify-content:space-between;">
                <span class="calib-label">{cat.upper()}</span>
                <span style="color:{pc};font-family:'Share Tech Mono',monospace;font-size:13px;">{pct}%</span>
              </div>
              <div class="accuracy-gauge"><div class="accuracy-fill" style="width:{pct}%;background:{pc};"></div></div>
              <div style="font-size:9px;color:#3a6080;">TP: {data['tp']} · FP: {data['fp']}</div>
            </div>
            """, unsafe_allow_html=True)

    # Feature importance
    importance = meta_clf.get_feature_importance()
    if importance:
        st.markdown('<div class="section-title">FEATURE IMPORTANCE</div>', unsafe_allow_html=True)
        max_imp = max(importance.values()) if importance.values() else 1
        html = ""
        for feat, imp in sorted(importance.items(), key=lambda x: -x[1]):
            bar_w = int((imp / max_imp) * 100) if max_imp > 0 else 0
            html += f"""<div class="feat-row">
              <span class="feat-name">{feat}</span>
              <div class="feat-bar-bg"><div class="feat-bar-fill" style="width:{bar_w}%;"></div></div>
              <span style="font-size:10px;color:#4a9eff;font-family:'Share Tech Mono',monospace;margin-left:6px;width:40px;">{imp:.3f}</span>
            </div>"""
        st.markdown(f'<div class="ml-status-box">{html}</div>', unsafe_allow_html=True)

    # ── Pending Reviews ──
    st.markdown(f'<div class="section-title">📋 PENDING REVIEWS ({len(pending)})</div>', unsafe_allow_html=True)

    if not pending:
        st.markdown("""<div style="color:#1e3a50;font-size:12px;font-family:'Share Tech Mono',monospace;
        letter-spacing:2px;padding:20px 0;text-align:center;">● NO PENDING REVIEWS</div>""", unsafe_allow_html=True)
        return

    for alert in pending:
        aid = alert["id"]
        signals = alert.get("signals", {})
        level = alert["threat_level"]
        tc_map = {"MEDIUM": "#ffe066", "HIGH": "#ff8c00", "CRITICAL": "#ff2255"}
        tc = tc_map.get(level, "#3a6080")

        # Card header
        st.markdown(f"""
        <div class="review-card" style="border-left:3px solid {tc};">
          <div class="review-header">
            <div>
              <span class="alert-level" style="color:{tc};">{level}</span>
              <span class="review-id">ID: {aid}</span>
            </div>
            <span class="alert-time">{alert.get('timestamp','')}</span>
          </div>
          <div class="alert-msg">{alert.get('reason','')}</div>
          <div class="review-signals">
            <div class="review-sig"><div class="review-sig-label">YOLO CONF</div><div class="review-sig-val">{signals.get('yolo_confidence',0):.2f}</div></div>
            <div class="review-sig"><div class="review-sig-label">FIGHT SCORE</div><div class="review-sig-val">{signals.get('fight_score',0):.3f}</div></div>
            <div class="review-sig"><div class="review-sig-label">MOTION</div><div class="review-sig-val">{signals.get('motion_score',0):.3f}</div></div>
            <div class="review-sig"><div class="review-sig-label">PEOPLE</div><div class="review-sig-val">{signals.get('people_count',0)}</div></div>
            <div class="review-sig"><div class="review-sig-label">ARM SPD</div><div class="review-sig-val">{signals.get('arm_speed',0):.3f}</div></div>
            <div class="review-sig"><div class="review-sig-label">PROXIMITY</div><div class="review-sig-val">{signals.get('proximity',0):.3f}</div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Snapshot
        snap = alert.get("snapshot")
        if snap and os.path.isfile(snap):
            st.image(snap, caption=f"Alert {aid}", use_column_width=True)

        # ML prediction
        ml_pred = alert.get("ml_prediction")
        if ml_pred is not None:
            ml_c = "#00ffa3" if ml_pred >= 0.5 else "#ff8c00"
            st.markdown(f"""<div style="font-size:11px;font-family:'Share Tech Mono',monospace;color:{ml_c};margin-bottom:6px;">
            🤖 ML CONFIDENCE: {ml_pred:.1%}</div>""", unsafe_allow_html=True)

        # Verdict buttons
        col_tp, col_fp = st.columns(2)
        if alert.get("threat_level") == "SAFE":
            with col_tp:
                if st.button("✅ TRUE SAFE", key=f"tn_{aid}"):
                    feedback_store.submit_feedback(aid, "true_negative")
                    _maybe_retrain(meta_clf, feedback_store)
                    st.rerun()
            with col_fp:
                if st.button("❌ MISSED THREAT", key=f"fn_{aid}"):
                    feedback_store.submit_feedback(aid, "false_negative")
                    _maybe_retrain(meta_clf, feedback_store)
                    st.rerun()
        else:
            with col_tp:
                if st.button("✅ TRUE THREAT", key=f"tp_{aid}"):
                    feedback_store.submit_feedback(aid, "true_positive")
                    _maybe_retrain(meta_clf, feedback_store)
                    st.rerun()
            with col_fp:
                if st.button("❌ FALSE ALARM", key=f"fp_{aid}"):
                    feedback_store.submit_feedback(aid, "false_positive")
                    _maybe_retrain(meta_clf, feedback_store)
                    st.rerun()

        st.markdown("---")

    # ── Recently Reviewed ──
    reviewed = feedback_store.get_reviewed(limit=10)
    if reviewed:
        st.markdown(f'<div class="section-title">✓ RECENTLY REVIEWED ({len(reviewed)})</div>', unsafe_allow_html=True)
        for r in reviewed:
            if r["verdict"] in ("true_positive", "false_negative"):
                vc = "#00ffa3" if r["verdict"] == "true_positive" else "#ff8c00"
                vt = "TRUE ✓" if r["verdict"] == "true_positive" else "MISSED ✗"
            else:
                vc = "#00ffa3" if r["verdict"] == "true_negative" else "#ff2255"
                vt = "SAFE ✓" if r["verdict"] == "true_negative" else "FALSE ✗"
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #0d2035;">
              <span style="font-size:11px;color:#3a6080;font-family:'Share Tech Mono',monospace;">{r.get('timestamp','')} · {r.get('threat_level','')}</span>
              <span style="font-size:11px;color:{vc};font-family:'Share Tech Mono',monospace;letter-spacing:2px;">{vt}</span>
            </div>""", unsafe_allow_html=True)


def _maybe_retrain(meta_clf, feedback_store):
    """Retrain meta-classifier if enough new data."""
    if meta_clf.should_retrain(feedback_store):
        with st.spinner("🧠 Training ML model..."):
            metrics = meta_clf.train(feedback_store)
            if metrics:
                st.success(f"Model trained! F1={metrics['f1']}, CV-F1={metrics['cv_f1']}")


def render_calibration_result(result):
    """Show auto-calibration results."""
    if result is None:
        st.warning("⚠ Not enough labelled data for calibration (need 15+)")
        return
    st.markdown(f"""
    <div class="ml-status-box">
      <div style="font-size:9px;color:#3a6080;letter-spacing:2px;font-family:'Share Tech Mono',monospace;">OPTIMAL PARAMETERS</div>
      <div class="calib-row"><span class="calib-label">YOLO CONFIDENCE</span><span class="calib-val">{result['yolo_conf']:.2f}</span></div>
      <div class="calib-row"><span class="calib-label">FIGHT THRESHOLD</span><span class="calib-val">{result['fight_threshold']:.2f}</span></div>
      <div class="calib-row"><span class="calib-label">MOTION SENSITIVITY</span><span class="calib-val">{result['motion_sensitivity']:.2f}</span></div>
      <div class="calib-row"><span class="calib-label">F1 SCORE</span><span class="calib-val">{result['f1_score']:.3f}</span></div>
      <div class="calib-row"><span class="calib-label">PRECISION</span><span class="calib-val">{result.get('precision',0):.3f}</span></div>
      <div class="calib-row"><span class="calib-label">RECALL</span><span class="calib-val">{result.get('recall',0):.3f}</span></div>
    </div>
    """, unsafe_allow_html=True)
