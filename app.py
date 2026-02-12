import json
import streamlit as st
from pathlib import Path

from src.pipeline import run_pipeline


st.set_page_config(page_title="Physician Notetaker", layout="wide")

st.title("🩺 Physician Notetaker — Medical NLP Demo")
st.caption("Paste a transcript → get Structured Summary, SOAP Note, Keywords, Sentiment + Intent")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Input Options")

use_sample = st.sidebar.checkbox("Use sample transcript", value=True)

transcript_text = ""

if use_sample:
    sample_path = Path("data/sample_transcript.txt")
    if sample_path.exists():
        transcript_text = sample_path.read_text(encoding="utf-8")
    else:
        st.sidebar.warning("Sample transcript not found at data/sample_transcript.txt")

uploaded = st.sidebar.file_uploader("Upload transcript (.txt)", type=["txt"])

if uploaded is not None:
    transcript_text = uploaded.read().decode("utf-8")
    use_sample = False

# -----------------------------
# Main Input
# -----------------------------
st.subheader("📄 Transcript Input")
transcript_text = st.text_area(
    "Paste transcript here:",
    value=transcript_text,
    height=350
)

# -----------------------------
# Run Pipeline
# -----------------------------
run_btn = st.button("🚀 Run Pipeline")

if run_btn:
    if not transcript_text.strip():
        st.error("Transcript is empty. Paste or upload a transcript.")
        st.stop()

    with st.spinner("Running pipeline..."):
        results = run_pipeline(transcript_text)

    st.success("Done!")

    # -----------------------------
    # Output Tabs
    # -----------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📌 Structured Summary",
        "🧾 SOAP Note",
        "😊 Sentiment + Intent",
        "🏷️ Keywords",
        "🧠 Model Summary"
    ])

    with tab1:
        st.json(results["structured_summary"])
        st.download_button(
            "Download medical_summary.json",
            data=json.dumps(results["structured_summary"], indent=2),
            file_name="medical_summary.json",
            mime="application/json"
        )

    with tab2:
        st.json(results["soap_note"])
        st.download_button(
            "Download soap_note.json",
            data=json.dumps(results["soap_note"], indent=2),
            file_name="soap_note.json",
            mime="application/json"
        )

    with tab3:
        st.json(results["sentiment_intent"])
        st.download_button(
            "Download sentiment_intent.json",
            data=json.dumps(results["sentiment_intent"], indent=2),
            file_name="sentiment_intent.json",
            mime="application/json"
        )

    with tab4:
        st.write(results["keywords"])
        st.download_button(
            "Download keywords.json",
            data=json.dumps({"keywords": results["keywords"]}, indent=2),
            file_name="keywords.json",
            mime="application/json"
        )

    with tab5:
        st.json(results["model_summary"])
        st.download_button(
            "Download model_summary.json",
            data=json.dumps(results["model_summary"], indent=2),
            file_name="model_summary.json",
            mime="application/json"
        )

    # -----------------------------
    # Optional: show full results
    # -----------------------------
    with st.expander("Show full pipeline output (debug)"):
        st.json(results)
