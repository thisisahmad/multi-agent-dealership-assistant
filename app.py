# app.py
# Streamlit UI for the streamlined 6-agent car dealership system

import json
import datetime
import os

import pytz
import streamlit as st

from car_dealer_agents import (
    run_orchestrator,
    build_kb_from_data,
    ensure_kb,
    get_available_dealers,
)

# ---------- Page config ----------
st.set_page_config(
    page_title="Car Dealership AI Assistant",
    page_icon="🚗",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --glass: rgba(255, 255, 255, 0.08);
        --glass-border: rgba(255, 255, 255, 0.16);
        --text-soft: rgba(255, 255, 255, 0.72);
        --accent: #7c3aed;
        --accent-2: #06b6d4;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        color: #f8fafc;
        background:
            radial-gradient(circle at top left, rgba(124, 58, 237, 0.35), transparent 34rem),
            radial-gradient(circle at top right, rgba(6, 182, 212, 0.24), transparent 30rem),
            linear-gradient(135deg, #050816 0%, #0b1020 45%, #111827 100%);
    }

    section[data-testid="stSidebar"] {
        background: rgba(10, 15, 30, 0.62);
        border-right: 1px solid var(--glass-border);
        backdrop-filter: blur(22px);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }

    h1 {
        font-size: clamp(2.2rem, 4vw, 4rem) !important;
        line-height: 1.05 !important;
        letter-spacing: -0.06em;
        background: linear-gradient(90deg, #ffffff 0%, #c4b5fd 45%, #67e8f9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    h3, .stMarkdown p {
        color: var(--text-soft);
    }

    div[data-testid="stChatMessage"] {
        border: 1px solid var(--glass-border);
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.11), rgba(255, 255, 255, 0.045));
        box-shadow: 0 22px 70px rgba(0, 0, 0, 0.26);
        backdrop-filter: blur(18px);
        padding: 0.75rem 1rem;
        margin: 0.85rem 0;
    }

    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.22), rgba(6, 182, 212, 0.10));
    }

    .stAlert, div[data-testid="stExpander"], div[data-testid="stInfo"] {
        border-radius: 18px;
        border: 1px solid var(--glass-border);
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(16px);
    }

    .stButton > button {
        width: 100%;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 16px;
        color: white;
        background: linear-gradient(135deg, var(--accent), var(--accent-2));
        box-shadow: 0 14px 32px rgba(6, 182, 212, 0.2);
        font-weight: 700;
        transition: transform 160ms ease, box-shadow 160ms ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 18px 42px rgba(124, 58, 237, 0.28);
    }

    div[data-testid="stChatInput"] {
        border-radius: 22px;
        border: 1px solid var(--glass-border);
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(18px);
    }

    div[data-baseweb="select"] > div {
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.08);
        border-color: var(--glass-border);
    }

    .hero-card {
        margin: 0.25rem 0 1.5rem;
        padding: 1.2rem 1.35rem;
        border-radius: 26px;
        border: 1px solid var(--glass-border);
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.11), rgba(255, 255, 255, 0.045));
        box-shadow: 0 24px 80px rgba(0, 0, 0, 0.28);
        backdrop-filter: blur(20px);
    }

    .hero-card strong {
        color: #ffffff;
    }

    .metric-pill {
        display: inline-flex;
        margin: 0.25rem 0.35rem 0.25rem 0;
        padding: 0.45rem 0.7rem;
        border-radius: 999px;
        color: #dbeafe;
        background: rgba(96, 165, 250, 0.12);
        border: 1px solid rgba(147, 197, 253, 0.22);
        font-size: 0.82rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar ----------
st.sidebar.title("🚗 Car Dealership AI")
st.sidebar.markdown("---")

# Dealer selection
available_dealers = get_available_dealers("Data")
if available_dealers:
    dealer_options = ["None (General KB)"] + available_dealers
    dealer_index = st.sidebar.selectbox(
        "Select Dealer (for dealer-specific RAG)",
        dealer_options,
        index=0,
        help="Select a dealer to use dealer-specific knowledge base. Answers will only come from that dealer's data."
    )
    dealer_name = None if dealer_index == "None (General KB)" else dealer_index
else:
    dealer_name = None
    st.sidebar.info("💡 Add PDF files to Data/. Embeddings saved in dealer folders based on filenames.")

st.sidebar.markdown("---")
st.sidebar.subheader("📚 Knowledge Base")

# Dealer-specific KB rebuild
if dealer_name:
    rebuild_label = f"🔄 Rebuild KB for {dealer_name}"
    if st.sidebar.button(rebuild_label):
        with st.spinner(f"Rebuilding FAISS KB for {dealer_name}..."):
            try:
                build_kb_from_data(dealer_name=dealer_name)
                st.sidebar.success(f"✅ KB rebuilt for {dealer_name}!")
            except Exception as e:
                st.sidebar.error(f"❌ Error: {str(e)}")
    
    # Make sure dealer-specific KB exists
    try:
        ensure_kb(dealer_name=dealer_name)
    except Exception as e:
        st.sidebar.warning(f"⚠️ KB for {dealer_name} not found. Please rebuild.")
else:
    if st.sidebar.button("🔄 Rebuild General KB from ./Data"):
        with st.spinner("Rebuilding FAISS KB from ./Data ..."):
            try:
                build_kb_from_data()
                st.sidebar.success("✅ KB rebuilt successfully!")
            except Exception as e:
                st.sidebar.error(f"❌ Error: {str(e)}")
    
    # Make sure general KB exists
    try:
        ensure_kb()
    except Exception as e:
        st.sidebar.warning(f"⚠️ KB not found. Please rebuild: {str(e)}")

if dealer_name:
    st.sidebar.info(f"💡 Using dealer-specific KB for: **{dealer_name}**")
    st.sidebar.info(f"📁 KB files: `Data/{dealer_name}/kb.index`")
    st.sidebar.info(f"📄 Documents stay in `Data/` folder (e.g., `{dealer_name}_*.pdf`)")
else:
    st.sidebar.info("💡 Add PDF files to Data/. Embeddings saved in dealer folders based on filenames.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 6 Core Agents")
st.sidebar.markdown("""
1. **Orchestrator** - Routes queries
2. **Intent Classification** - Classifies & confirms intent
3. **Form Filling** - Collects form data (one question at a time)
4. **QA Agent** - Validates forms before finalizing
5. **Knowledge** - RAG-powered FAQs (dealer-specific)
6. **Handoff** - Human escalation
""")

# ---------- Main area ----------
st.title("🚗 Car Dealership AI Assistant")
st.markdown(
    """
    <div class="hero-card">
        <strong>Premium multi-agent dealership assistant</strong><br/>
        Ask dealer-specific questions, build bookings from natural conversation, and keep every answer grounded in the selected knowledge base.
        <div style="margin-top: 0.85rem;">
            <span class="metric-pill">Dealer-specific RAG</span>
            <span class="metric-pill">Natural date & time booking</span>
            <span class="metric-pill">GPT-5 reasoning</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_logs" not in st.session_state:
    st.session_state.agent_logs = []

if st.sidebar.button("🧹 Clear chat"):
    st.session_state.messages = []
    st.session_state.agent_logs = []
    st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about services, book appointments, or get information..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response, logs = run_orchestrator(
                    prompt,
                    orgName="Demo Motors",
                    tz="Europe/London",
                    language="en",
                    dealer_name=dealer_name,
                )
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.agent_logs = logs
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Agent logs
if st.session_state.agent_logs:
    with st.expander("🔍 Agent Trace (Click to view)"):
        for log_entry in st.session_state.agent_logs:
            st.json(log_entry)

