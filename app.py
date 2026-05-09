"""
DeepFake Shield — Main Streamlit Application
=============================================
Run with:  streamlit run app.py
"""

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from detector.image_detector import analyze_image
from detector.video_detector import analyze_video
from detector.model_manager  import model_manager, DEFAULT_MODEL_ID
from utils.file_utils import (
    cleanup_file,
    human_size,
    save_uploaded_file,
    validate_file,
)
from utils.report_utils import generate_report, save_report


# ── Cached model loader (loaded once per Streamlit session) ───────────────────
@st.cache_resource(show_spinner=False)
def load_model_cached(model_id: str) -> tuple[bool, str]:
    """Download / load the model and return (success, error_message)."""
    ok = model_manager.load(model_id)
    return ok, (model_manager.load_error or "")

# ── Ensure required directories exist ─────────────────────────────────────────
Path("uploads").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepFake Shield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Header banner ─────────────────────────────────────────────────────── */
.shield-header {
    background: linear-gradient(135deg, #0d1b2a 0%, #1b3a5c 60%, #1e5799 100%);
    color: white;
    padding: 28px 24px 20px;
    border-radius: 14px;
    text-align: center;
    margin-bottom: 24px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.25);
}
.shield-header h1 { margin: 0; font-size: 2.4rem; letter-spacing: 1px; }
.shield-header p  { margin: 6px 0 0; opacity: 0.85; font-size: 1rem; }

/* ── Verdict badges ────────────────────────────────────────────────────── */
.verdict-badge {
    display: inline-block;
    padding: 10px 28px;
    border-radius: 24px;
    font-weight: 700;
    font-size: 1.25rem;
    letter-spacing: 0.5px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.18);
}
.v-real        { background: #1a7a3c; color: #fff; }
.v-suspicious  { background: #c05e00; color: #fff; }
.v-ai          { background: #b5161e; color: #fff; }
.v-inconclusive{ background: #555;    color: #fff; }

/* ── Info card ─────────────────────────────────────────────────────────── */
.info-card {
    background: #f0f4fa;
    border-left: 4px solid #1e5799;
    padding: 14px 18px;
    border-radius: 0 10px 10px 0;
    margin: 10px 0;
    font-size: 0.95rem;
}

/* ── Disclaimer ────────────────────────────────────────────────────────── */
.disclaimer {
    background: #fffbe6;
    border: 1px solid #e6c100;
    border-left: 5px solid #e6c100;
    padding: 14px 18px;
    border-radius: 0 10px 10px 0;
    margin: 18px 0;
    font-size: 0.88rem;
    color: #5a4a00;
}

/* ── Privacy notice ────────────────────────────────────────────────────── */
.privacy {
    background: #e3f4f8;
    border-left: 4px solid #0c9ab8;
    padding: 10px 14px;
    border-radius: 0 8px 8px 0;
    font-size: 0.82rem;
    color: #0a5f73;
    margin: 10px 0;
}

/* ── Signal bar label ──────────────────────────────────────────────────── */
.sig-label { font-size: 0.85rem; color: #444; margin: 0; }
</style>
""",
    unsafe_allow_html=True,
)


# ── Session state ─────────────────────────────────────────────────────────────
if "scan_history" not in st.session_state:
    st.session_state.scan_history: list[dict] = []


# ── Helpers ───────────────────────────────────────────────────────────────────

def _verdict_css(verdict: str) -> str:
    return {
        "Likely Real":             "v-real",
        "Suspicious":              "v-suspicious",
        "Likely AI / Manipulated": "v-ai",
        "Inconclusive":            "v-inconclusive",
    }.get(verdict, "v-inconclusive")


def _verdict_icon(verdict: str) -> str:
    return {
        "Likely Real":             "✅",
        "Suspicious":              "⚠️",
        "Likely AI / Manipulated": "🚨",
        "Inconclusive":            "❓",
    }.get(verdict, "❓")


def _prob_color(prob: float) -> str:
    if prob < 0.32:
        return "#1a7a3c"
    elif prob < 0.52:
        return "#666"
    elif prob < 0.72:
        return "#c05e00"
    return "#b5161e"


def display_results(report: dict, file_type: str) -> None:
    """Render the full analysis results section."""

    verdict  = report["verdict"]
    css      = _verdict_css(verdict)
    icon     = _verdict_icon(verdict)
    ai_prob  = report["ai_probability"]
    conf     = report["confidence_level"]

    # ── Verdict banner ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        f'<div style="text-align:center;padding:18px 0;">'
        f'<div class="verdict-badge {css}">{icon} {verdict}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Model badge ───────────────────────────────────────────────────────
    model_id = report.get("technical_details", {}).get("model_id")
    if model_id:
        st.caption(f"Neural model: `{model_id}` · Ensemble mode (70% model + 30% heuristics)")
    else:
        st.caption("Heuristics-only mode — load the neural model in the sidebar for higher accuracy")

    # ── Core metric tiles ─────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        color = _prob_color(ai_prob)
        st.markdown(
            f'<div style="text-align:center;">'
            f'<div style="font-size:2.2rem;font-weight:700;color:{color};">'
            f'{ai_prob:.0%}</div>'
            f'<div style="color:#666;font-size:0.85rem;">AI / Fake Probability</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div style="text-align:center;">'
            f'<div style="font-size:2.2rem;font-weight:700;color:#1e5799;">'
            f'{conf:.0%}</div>'
            f'<div style="color:#666;font-size:0.85rem;">Confidence</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div style="text-align:center;">'
            f'<div style="font-size:2.2rem;font-weight:700;color:#333;">'
            f'{file_type.capitalize()}</div>'
            f'<div style="color:#666;font-size:0.85rem;">Media Type</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Left / Right columns ──────────────────────────────────────────────
    left, right = st.columns([3, 2])

    with left:
        st.markdown("#### Explanation")
        st.info(report["explanation"])

        if report["suspicious_signals"]:
            st.markdown("#### Suspicious Signals")
            for sig in report["suspicious_signals"]:
                st.markdown(f"- {sig}")
        else:
            st.success("No strong suspicious signals detected in this media.")

        # Video: suspicious frames table
        if file_type == "video" and report.get("suspicious_frames"):
            st.markdown("#### Suspicious Frames")
            df = pd.DataFrame(report["suspicious_frames"])
            st.dataframe(df, use_container_width=True, hide_index=True)
        elif file_type == "video":
            st.success("No suspicious frames detected across the sampled timeline.")

    with right:
        st.markdown("#### Detection Signal Breakdown")
        st.caption(
            "Each bar shows the risk contribution of one detection heuristic. "
            "Bars above 50% are highlighted."
        )

        for sig_name, sig_val in report["signal_breakdown"].items():
            pct = int(sig_val * 100)
            bar_color = "#b5161e" if sig_val >= 0.52 else "#1e5799"
            st.markdown(
                f'<p class="sig-label"><b>{sig_name}</b> — {pct}%</p>',
                unsafe_allow_html=True,
            )
            st.progress(float(sig_val))

        # Video: frame score timeline
        if file_type == "video" and report.get("frame_scores"):
            with st.expander("Frame-by-frame score timeline"):
                scores = report["frame_scores"]
                chart_data = pd.DataFrame(
                    {"Frame Score": scores},
                    index=[f"F{i+1}" for i in range(len(scores))],
                )
                st.line_chart(chart_data)

        with st.expander("Technical Details"):
            st.json(report.get("technical_details", {}))

    # ── Grad-CAM Explainability ───────────────────────────────────────────
    if file_type == "image" and model_manager.is_loaded:
        st.markdown("---")
        st.markdown("#### 🔬 Explainability Map (Grad-CAM)")
        st.caption(
            "Grad-CAM shows **which regions** the neural model focused on. "
            "Red = most suspicious, Blue = least suspicious. "
            "Method: gradient-weighted class activation mapping on the last ViT encoder layer."
        )

        _file_path = report.get("_tmp_path_for_gradcam")   # set by caller if available
        if _file_path:
            with st.spinner("Generating heatmap…"):
                from detector.explainability import generate_explanation
                xai = generate_explanation(_file_path)

            if xai["error"]:
                st.warning(f"Explainability unavailable: {xai['error']}")
            else:
                col_orig, col_heat = st.columns(2)
                with col_orig:
                    st.image(xai["original"], caption="Original (resized to model input)", use_container_width=True)
                with col_heat:
                    st.image(xai["overlay"],  caption=f"Grad-CAM overlay ({xai['method']})", use_container_width=True)

                st.caption(
                    f"Most suspicious region: **{xai['top_region']}**  |  "
                    f"Model fake score: **{xai['fake_score']:.1%}**  |  "
                    f"Method: **{xai['method']}**"
                )
        else:
            st.info(
                "Explainability map is generated during live analysis. "
                "Upload a new image to see the heatmap."
            )
    elif file_type == "image" and not model_manager.is_loaded:
        st.info("Load the neural model from the sidebar to enable Grad-CAM explainability maps.")

    # ── Disclaimer ────────────────────────────────────────────────────────
    st.markdown(
        '<div class="disclaimer">'
        "<b>⚠️ Disclaimer:</b> This analysis is a probabilistic risk estimate produced by "
        "statistical heuristics — <b>not</b> a trained forensic model. It is <b>not</b> "
        "admissible as legal or forensic evidence. Results may be inaccurate. Always "
        "consult qualified experts before acting on this output."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Export ────────────────────────────────────────────────────────────
    st.download_button(
        label="⬇️ Download Report (JSON)",
        data=json.dumps(report, indent=2, ensure_ascii=False),
        file_name=f"deepfake_shield_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ DeepFake Shield")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Analyze", "History", "About"],
        label_visibility="collapsed",
    )

    # ── Neural model panel ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🤖 Neural Model")

    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False

    if model_manager.is_loaded:
        st.success(f"Loaded")
        st.caption(f"`{model_manager.model_id}`")
    else:
        st.info("Not loaded — using heuristics only")
        custom_id = st.text_input(
            "Model ID (HuggingFace)",
            value=DEFAULT_MODEL_ID,
            label_visibility="collapsed",
            placeholder=DEFAULT_MODEL_ID,
        )
        if st.button("⬇️ Load Model", use_container_width=True):
            with st.spinner("Downloading model… (first time ~85–300 MB)"):
                ok, err = load_model_cached(custom_id.strip() or DEFAULT_MODEL_ID)
            if ok:
                st.session_state.model_loaded = True
                st.rerun()
            else:
                st.error(f"Load failed:\n{err}")

    st.markdown("---")
    st.markdown(
        '<div class="privacy">'
        "🔒 <b>Privacy First</b><br>"
        "Uploaded files are written to a temporary local folder, analyzed, "
        "then deleted immediately. No media leaves your machine."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    if st.session_state.scan_history:
        st.markdown(
            f"**{len(st.session_state.scan_history)}** scan(s) in this session"
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYZE
# ══════════════════════════════════════════════════════════════════════════════
if page == "Analyze":

    st.markdown(
        '<div class="shield-header">'
        "<h1>🛡️ DeepFake Shield</h1>"
        "<p>AI-Powered Media Authenticity Analysis — Upload an image or video to begin.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    tab_upload, tab_url = st.tabs(["📁 Upload File", "🔗 Paste URL"])

    uploaded_file = None

    with tab_upload:
        st.markdown(
            '<div class="info-card">'
            "Supported formats: <b>JPG, JPEG, PNG</b> (images) · "
            "<b>MP4, MOV, AVI</b> (videos) · Max size: <b>200 MB</b>"
            "</div>",
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["jpg", "jpeg", "png", "mp4", "mov", "avi"],
            label_visibility="collapsed",
        )

    with tab_url:
        st.markdown(
            '<div class="info-card">'
            "Paste a media URL below. For this MVP, please download the file and "
            "upload it via the <b>Upload File</b> tab. Direct URL fetching will be "
            "added in the next release."
            "</div>",
            unsafe_allow_html=True,
        )
        url_input = st.text_input(
            "Media URL", placeholder="https://example.com/photo.jpg"
        )
        if url_input:
            st.info(
                "URL saved. To analyze this media, download the file and upload it "
                "using the Upload File tab."
            )

    # ── File uploaded ─────────────────────────────────────────────────────
    if uploaded_file is not None:

        is_valid, err_msg, file_type = validate_file(uploaded_file)

        if not is_valid:
            st.error(f"**Validation failed:** {err_msg}")
            st.stop()

        col_preview, col_info = st.columns([1, 1])

        # Save file to disk so both preview and analysis can use it
        file_path = save_uploaded_file(uploaded_file, "uploads")

        with col_preview:
            st.markdown("### Preview")
            if file_type == "image":
                st.image(file_path, use_container_width=True, caption=uploaded_file.name)
            else:
                st.video(file_path)

        with col_info:
            st.markdown("### File Information")
            size_str = human_size(uploaded_file.size)
            st.markdown(f"**Filename:** {uploaded_file.name}")
            st.markdown(f"**Type:** {file_type.capitalize()}")
            st.markdown(f"**Size:** {size_str}")
            st.markdown(f"**Format:** {uploaded_file.type or 'unknown'}")

            st.markdown("---")

            if file_type == "video":
                st.caption(
                    "Video analysis samples up to 12 frames and checks temporal consistency. "
                    "Processing may take 10–30 seconds."
                )

            analyze_btn = st.button(
                "🔍 Analyze for Deepfakes",
                type="primary",
                use_container_width=True,
            )

        # ── Run analysis ──────────────────────────────────────────────────
        if analyze_btn:
            status = st.status("Running DeepFake Shield analysis…", expanded=True)

            with status:
                st.write("Validating file…")
                if file_type == "image":
                    st.write("Extracting image signals…")
                    result = analyze_image(file_path)
                else:
                    st.write("Extracting video frames…")
                    st.write("Analyzing frames for anomalies…")
                    st.write("Checking temporal consistency…")
                    result = analyze_video(file_path)

                if result.get("error"):
                    status.update(label="Analysis failed", state="error")
                    st.error(result["error"])
                    cleanup_file(file_path)
                    st.stop()

                st.write("Generating report…")
                report = generate_report(
                    filename=uploaded_file.name,
                    file_type=file_type,
                    result=result,
                )
                report_path = save_report(report)

                # Add to session history
                st.session_state.scan_history.append(
                    {
                        "ID": f"SCAN-{len(st.session_state.scan_history)+1:04d}",
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Filename": uploaded_file.name,
                        "Type": file_type.capitalize(),
                        "Verdict": report["verdict"],
                        "AI Probability": f'{report["ai_probability"]:.0%}',
                        "Confidence": f'{report["confidence_level"]:.0%}',
                        "_report": report,  # full report for detail view
                    }
                )

            status.update(label="Analysis complete", state="complete", expanded=False)

            # Pass temp path so display_results can run Grad-CAM before cleanup
            report["_tmp_path_for_gradcam"] = file_path if file_type == "image" else None
            display_results(report, file_type)
            report.pop("_tmp_path_for_gradcam", None)   # strip before saving

            # Cleanup upload after analysis
            cleanup_file(file_path)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "History":

    st.markdown("## Scan History")
    st.caption("History is stored for the current session only and resets on restart.")

    if not st.session_state.scan_history:
        st.info("No scans yet. Go to **Analyze** to scan your first file.")
    else:
        # Summary table (hide internal _report column)
        display_cols = ["ID", "Timestamp", "Filename", "Type", "Verdict",
                        "AI Probability", "Confidence"]
        df = pd.DataFrame(st.session_state.scan_history)[display_cols]
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### View Scan Detail")

        scan_ids = [s["ID"] for s in st.session_state.scan_history]
        selected = st.selectbox("Select a scan", scan_ids, label_visibility="collapsed")

        if selected:
            entry = next(
                s for s in st.session_state.scan_history if s["ID"] == selected
            )
            full_report = entry["_report"]

            st.markdown(f"**File:** {entry['Filename']}  |  **Scanned:** {entry['Timestamp']}")

            col_dl, _ = st.columns([1, 3])
            with col_dl:
                st.download_button(
                    "⬇️ Download Report",
                    data=json.dumps(full_report, indent=2),
                    file_name=f"{selected}.json",
                    mime="application/json",
                )

            with st.expander("Full report JSON"):
                st.json(full_report)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "About":

    st.markdown("## About DeepFake Shield")

    st.markdown(
        """
DeepFake Shield is an **open-source, privacy-first** media authenticity checker
built as a final-year engineering project. It is designed to be educational and
modular — the detection backend can be upgraded to any real AI model without
touching the UI or reporting logic.

---
### How It Works

| Stage | What happens |
|---|---|
| **Upload** | File is validated (type, size) and written to a local temp folder. |
| **Image analysis** | 6 statistical signals are computed (metadata, noise, color, ELA, edges, FFT). |
| **Video analysis** | Up to 12 frames are sampled, analyzed individually, then temporal consistency is checked. |
| **Scoring** | Signals are combined into a weighted probability score (0–100%). |
| **Verdict** | Score mapped to: *Likely Real / Inconclusive / Suspicious / Likely AI*. |
| **Cleanup** | Upload is deleted immediately after analysis. |

---
### Current Detection Signals

| Signal | Basis |
|---|---|
| **Metadata Integrity** | Missing EXIF / AI tool names in metadata |
| **Noise Pattern** | Local noise uniformity vs camera sensor noise |
| **Color Distribution** | Histogram smoothness anomalies |
| **Compression Artifacts** | Error Level Analysis (ELA) |
| **Edge Consistency** | Canny edge density and ratio |
| **Frequency Artifacts** | FFT mid-frequency variance (GAN grid artifacts) |
| **Temporal Consistency** | Frame-score variance across sampled video frames |

---
### Limitations

- **No trained model** — current signals are heuristics, not learned from deepfake data.
- **False positive rate** — some authentic images will score as suspicious.
- **Not forensic-grade** — do not use results as evidence.
- **Images only, no audio** — audio-based cloning detection is not yet implemented.

---
### Roadmap (Post-MVP)

1. Plug in a pre-trained model (FaceForensics++, CNNDetection, CLIP-based)
2. Face-region cropping + face-specific deepfake classifier
3. Audio-visual sync analysis for videos
4. REST API (FastAPI) for programmatic access
5. Batch upload and comparison mode
6. PDF report export

---
### Tech Stack
`Python` · `Streamlit` · `OpenCV` · `Pillow` · `NumPy` · `Pandas`

---
### Disclaimer
This tool provides probabilistic risk estimates only. It is not forensic proof and
should not be used for legal, journalistic, or law-enforcement decisions. Always
consult qualified experts for critical determinations.
"""
    )
