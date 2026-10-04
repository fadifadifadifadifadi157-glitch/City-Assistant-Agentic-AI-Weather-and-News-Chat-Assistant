import os
import uuid
import requests
from dotenv import load_dotenv

import streamlit as st
import streamlit.components.v1 as components

from langchain_core.tools import tool
from langchain_core.messages import ToolMessage

from langchain_groq import ChatGroq
from tavily import TavilyClient
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="City Assistant",
    page_icon="🌆",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLING — MODERN SAAS DARK (gradients, glass, motion)
# ============================================================

st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">

    <style>
    :root {
        --accent-indigo: #818cf8;
        --accent-violet: #a78bfa;
        --accent-cyan: #38bdf8;
        --accent-green: #4ade80;
        --accent-amber: #fbbf24;
        --accent-rose: #fb7185;
        --bg-deep: #0b0b14;
        --panel: #14141f;
        --panel-2: #1a1a28;
        --border-subtle: rgba(255,255,255,0.09);
        --border-glow: rgba(129,140,248,0.35);
        --text-primary: #eef0f6;
        --text-muted: #9497ac;
    }

    html, body, .stApp {
        background:
            radial-gradient(circle at 15% 0%, rgba(129,140,248,0.14) 0%, transparent 45%),
            radial-gradient(circle at 85% 20%, rgba(56,189,248,0.10) 0%, transparent 45%),
            radial-gradient(circle at 50% 100%, rgba(167,139,250,0.10) 0%, transparent 50%),
            var(--bg-deep) !important;
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }

    /* ---- KEYFRAMES ---- */
    @keyframes fadeInUp {
        0% { opacity: 0; transform: translateY(14px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes softPulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(251,191,36,0.28); }
        50% { box-shadow: 0 0 0 8px rgba(251,191,36,0); }
    }
    @keyframes floatIcon {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-3px); }
    }
    @keyframes shimmerLine {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }

    /* ---- HEADER ---- */
    .main-header {
        text-align: center;
        padding: 1.7rem 1rem 1.5rem;
        margin-bottom: 1rem;
        background: linear-gradient(145deg, rgba(129,140,248,0.10), rgba(56,189,248,0.06) 60%, rgba(26,26,40,0.4));
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04);
        position: relative;
        overflow: hidden;
        animation: fadeInUp 0.55s ease both;
    }
    .main-header::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--accent-indigo), var(--accent-cyan), var(--accent-violet), var(--accent-indigo));
        background-size: 300% 100%;
        animation: gradientShift 6s ease infinite;
    }
    .main-header h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        margin: 0;
        font-size: 1.8rem;
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
    }
    .main-header .header-icon {
        font-size: 1.9rem;
        line-height: 1;
        animation: floatIcon 2.6s ease-in-out infinite;
        filter: drop-shadow(0 0 10px rgba(129,140,248,0.45));
    }
    .main-header .header-text {
        background: linear-gradient(90deg, #ffffff 0%, var(--accent-indigo) 45%, var(--accent-cyan) 75%, #ffffff 100%);
        background-size: 300% 100%;
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        animation: gradientShift 7s ease infinite;
    }
    .main-header p {
        color: var(--text-muted);
        margin: 0.4rem 0 0 0;
        font-size: 0.84rem;
        font-weight: 500;
        letter-spacing: 0.4px;
    }

    .sidebar-hint {
        text-align: center;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.76rem;
        color: var(--accent-cyan);
        padding: 9px 12px;
        margin-bottom: 1rem;
        border: 1px solid rgba(56,189,248,0.25);
        border-radius: 10px;
        background: rgba(56,189,248,0.06);
        animation: fadeInUp 0.55s ease both;
        animation-delay: 0.05s;
    }

    /* ---- STAT DASHBOARD CARDS ---- */
    .stat-card {
        border-radius: 14px;
        padding: 12px 12px 10px;
        text-align: center;
        margin-bottom: 0.6rem;
        min-height: 78px;
        width: 100%;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        border: 1px solid var(--border-subtle);
        background: linear-gradient(160deg, var(--panel-2) 0%, var(--panel) 100%);
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
        position: relative;
        overflow: hidden;
        animation: fadeInUp 0.5s ease both;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }
    .stat-card:hover {
        transform: translateY(-3px);
        border-color: var(--border-glow);
        box-shadow: 0 10px 28px rgba(0,0,0,0.45);
    }
    .stat-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
    }
    .stat-cyan::before    { background: linear-gradient(90deg, var(--accent-cyan), #0ea5e9); }
    .stat-green::before   { background: linear-gradient(90deg, var(--accent-green), #16a34a); }
    .stat-magenta::before { background: linear-gradient(90deg, var(--accent-violet), #c026d3); }
    .stat-yellow::before  { background: linear-gradient(90deg, var(--accent-amber), #f97316); }
    .stat-white::before   { background: linear-gradient(90deg, #ffffff, #cbd5e1); }

    .stat-card .stat-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.63rem;
        letter-spacing: 1.4px;
        color: var(--text-muted);
        text-transform: uppercase;
    }
    .stat-card .stat-value {
        font-family: 'Inter', sans-serif;
        font-size: 1.45rem;
        font-weight: 800;
        margin-top: 4px;
    }
    .stat-cyan    .stat-value { color: var(--accent-cyan); }
    .stat-magenta .stat-value { color: var(--accent-violet); }
    .stat-green   .stat-value { color: var(--accent-green); }
    .stat-yellow  .stat-value { color: var(--accent-amber); }
    .stat-white   .stat-value { color: #ffffff; }

    /* ---- CHAT MESSAGES ---- */
    [data-testid="stChatMessage"] {
        border-radius: 14px;
        margin-bottom: 0.65rem;
        animation: fadeInUp 0.4s ease both;
    }

    div[data-testid="stChatMessageContent"] {
        border-radius: 14px;
        padding: 14px 16px;
        background: linear-gradient(160deg, var(--panel-2), var(--panel));
        font-family: 'Inter', sans-serif;
        box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    }

    .stChatMessage:has(div[data-testid="stChatMessageAvatarUser"]) div[data-testid="stChatMessageContent"] {
        border: 1px solid rgba(167,139,250,0.35);
        color: var(--text-primary);
    }

    .stChatMessage:has(div[data-testid="stChatMessageAvatarAssistant"]) div[data-testid="stChatMessageContent"] {
        border: 1px solid rgba(56,189,248,0.28);
        color: var(--text-primary);
    }

    /* ---- SIDEBAR ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--panel) 0%, var(--bg-deep) 100%);
        border-right: 1px solid var(--border-subtle);
        font-family: 'Inter', sans-serif;
    }

    section[data-testid="stSidebar"] * {
        color: var(--text-primary) !important;
    }

    section[data-testid="stSidebar"] h2 {
        background: linear-gradient(90deg, var(--accent-indigo), var(--accent-cyan));
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent !important;
        font-weight: 800;
    }

    section[data-testid="stSidebar"] div.stButton > button {
        background: rgba(255,255,255,0.03) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 10px !important;
        transition: all 0.18s ease;
    }

    section[data-testid="stSidebar"] div.stButton > button:hover {
        border-color: var(--border-glow) !important;
        background: rgba(129,140,248,0.08) !important;
    }

    /* ---- PERMISSION REQUEST PANEL ---- */
    .approval-box {
        border: 1px solid rgba(251,191,36,0.45);
        border-radius: 14px;
        padding: 18px 20px;
        background: linear-gradient(160deg, rgba(251,191,36,0.07), var(--panel));
        margin-bottom: 0.6rem;
        animation: fadeInUp 0.4s ease both, softPulse 2.4s ease-in-out infinite;
    }

    .approval-title {
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        color: var(--accent-amber);
        font-weight: 700;
        letter-spacing: 0.2px;
        margin-bottom: 14px;
        display: inline-block;
        animation: floatIcon 2.2s ease-in-out infinite;
    }

    .perm-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 9px 0;
        border-bottom: 1px solid var(--border-subtle);
    }
    .perm-row:last-of-type {
        border-bottom: none;
    }

    .perm-key {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.71rem;
        letter-spacing: 1.2px;
        color: var(--text-muted);
    }

    .perm-val {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        color: var(--accent-cyan);
        background: rgba(56,189,248,0.1);
        padding: 5px 13px;
        border-radius: 7px;
        border: 1px solid rgba(56,189,248,0.3);
    }

    .approval-status {
        margin-top: 14px;
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.3px;
        font-size: 0.8rem;
        color: var(--accent-green);
        background: linear-gradient(90deg, var(--accent-green) 0%, rgba(74,222,128,0.3) 50%, var(--accent-green) 100%);
        background-size: 200% 100%;
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmerLine 2.5s linear infinite;
    }

    /* ---- BUTTONS ---- */
    div.stButton > button {
        border-radius: 10px;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        letter-spacing: 0.2px;
        border: none;
        padding: 12px;
        transition: all 0.18s ease-in-out;
        cursor: pointer;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
    }

    div.stButton > button:active {
        transform: translateY(0px);
    }

    div.stButton > button[kind="primary"],
    div.stButton > button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, var(--accent-green), #16a34a) !important;
        color: #052e1a !important;
        border: none !important;
        box-shadow: 0 4px 16px rgba(74,222,128,0.28);
    }

    div.stButton > button[kind="primary"]:hover,
    div.stButton > button[data-testid="stBaseButton-primary"]:hover {
        box-shadow: 0 6px 22px rgba(74,222,128,0.4);
        filter: brightness(1.05);
    }

    div.stButton > button[kind="secondary"],
    div.stButton > button[data-testid="stBaseButton-secondary"] {
        background: rgba(251,113,133,0.08) !important;
        color: var(--accent-rose) !important;
        border: 1px solid rgba(251,113,133,0.4) !important;
    }

    div.stButton > button[kind="secondary"]:hover,
    div.stButton > button[data-testid="stBaseButton-secondary"]:hover {
        background: rgba(251,113,133,0.16) !important;
        box-shadow: 0 6px 18px rgba(251,113,133,0.22);
    }

    /* ---- CHAT INPUT ---- */
    [data-testid="stChatInput"] {
        border-radius: 12px !important;
        border: 1px solid var(--border-subtle) !important;
        background: linear-gradient(160deg, var(--panel-2), var(--panel)) !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: var(--border-glow) !important;
        box-shadow: 0 0 0 3px rgba(129,140,248,0.15) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TOOLS (unchanged logic from Agent2.py)
# ============================================================

@tool
def get_weather(city: str) -> str:
    """Get the current weather of a city in Pakistan."""

    if not OPENWEATHER_API_KEY:
        return "OpenWeather API key is missing."

    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": f"{city},PK",
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        return f"OpenWeather error {response.status_code}: {response.text}"

    data = response.json()

    return str({
        "city": data["name"],
        "temperature": data["main"]["temp"],
        "feels_like": data["main"]["feels_like"],
        "humidity": data["main"]["humidity"],
        "description": data["weather"][0]["description"],
        "wind_speed": data["wind"]["speed"]
    })


tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


@tool
def search_city_news(city: str) -> str:
    """Search for the latest news about a specific city in Pakistan."""

    response = tavily_client.search(
        query=f"latest news about {city} Pakistan",
        max_results=2
    )

    results = response.get("results", [])

    if not results:
        return f"No latest news found for {city}."

    output = []

    for result in results:
        title = result.get("title", "No title")
        content = result.get("content", "No content")
        content = content[:500]

        output.append(f"Title: {title}\nContent: {content}")

    return "\n\n".join(output)


# ============================================================
# AGENT (with human-approval middleware adapted for Streamlit)
# ============================================================

@wrap_tool_call
def humanApproval(request, handler):
    """Pauses the agent and asks a human (via the UI) to approve the tool call."""

    tool_name = request.tool_call["name"]
    tool_args = request.tool_call["args"]

    decision = interrupt({"tool_name": tool_name, "tool_args": tool_args})

    if str(decision).lower() != "yes":
        return ToolMessage(
            content="The user denied this tool call.",
            tool_call_id=request.tool_call["id"]
        )

    return handler(request)


@st.cache_resource(show_spinner=False)
def build_agent():
    llm = ChatGroq(model="openai/gpt-oss-120b")

    checkpointer = MemorySaver()

    agent = create_agent(
        llm,
        tools=[get_weather, search_city_news],
        system_prompt=(
            "You are a helpful city assistant. "
            "Give short and direct answers when the user asks for short answers."
        ),
        middleware=[humanApproval],
        checkpointer=checkpointer,
    )
    return agent


agent = build_agent()


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "pending_approval" not in st.session_state:
    st.session_state.pending_approval = None

if "approved_count" not in st.session_state:
    st.session_state.approved_count = 0

if "denied_count" not in st.session_state:
    st.session_state.denied_count = 0

if "mode" not in st.session_state:
    st.session_state.mode = "Manual"

config = {"configurable": {"thread_id": st.session_state.thread_id}}


def run_agent(input_data):
    """Invoke the agent and update session state with the result or a pending approval.

    In Manual mode, the first tool-call interrupt stops execution and shows the
    approval card. In Auto mode, every interrupt is immediately resumed with
    'yes' so the agent runs straight through to a final answer.
    """

    result = agent.invoke(input_data, config=config)

    while isinstance(result, dict) and result.get("__interrupt__"):
        if st.session_state.mode == "Auto":
            st.session_state.approved_count += 1
            result = agent.invoke(Command(resume="yes"), config=config)
        else:
            interrupt_obj = result["__interrupt__"][0]
            st.session_state.pending_approval = interrupt_obj.value
            return

    final_message = result["messages"][-1].content
    st.session_state.messages.append({"role": "assistant", "content": final_message})
    st.session_state.pending_approval = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🌆 CITY ASSISTANT")
    st.markdown("_> weather &amp; news uplink for Pakistan_")
    st.markdown("---")

    st.markdown("**[ API KEY STATUS ]**")
    st.markdown(f"- OpenWeather: {'ONLINE' if OPENWEATHER_API_KEY else 'OFFLINE'}")
    st.markdown(f"- Tavily: {'ONLINE' if TAVILY_API_KEY else 'OFFLINE'}")

    st.markdown("---")

    st.markdown("**[ EXECUTION MODE ]**")
    mode_col1, mode_col2 = st.columns(2)

    with mode_col1:
        manual_clicked = st.button(
            "MANUAL",
            use_container_width=True,
            type="primary" if st.session_state.mode == "Manual" else "secondary",
        )

    with mode_col2:
        auto_clicked = st.button(
            "AUTO",
            use_container_width=True,
            type="primary" if st.session_state.mode == "Auto" else "secondary",
        )

    if manual_clicked and st.session_state.mode != "Manual":
        st.session_state.mode = "Manual"
        st.rerun()

    if auto_clicked and st.session_state.mode != "Auto":
        st.session_state.mode = "Auto"
        st.rerun()

    st.caption("Manual = approve every tool call. Auto = runs straight through.")

    st.markdown("---")

    if st.button("🔄 RESET CHAT", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.pending_approval = None
        st.session_state.approved_count = 0
        st.session_state.denied_count = 0
        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-header">
        <h1><span class="header-icon">🌆</span><span class="header-text">CITY ASSISTANT</span></h1>
        <p>&gt;&gt; WEATHER &amp; NEWS UPLINK // PAKISTAN &lt;&lt;</p>
    </div>
    <div class="sidebar-hint">
        ⬅ Open the sidebar (click <b>&raquo;</b> top-left) to switch <b>MODE</b> (Manual/Auto) or <b>RESET CHAT</b>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DASHBOARD — STAT CARDS
# ============================================================

stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns(5)

with stat_col1:
    st.markdown(
        f"""
        <div class="stat-card stat-cyan" style="animation-delay:0s, 0s;">
            <div class="stat-label">MESSAGES</div>
            <div class="stat-value">{len(st.session_state.messages)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with stat_col2:
    st.markdown(
        f"""
        <div class="stat-card stat-green" style="animation-delay:0.1s, 0.2s;">
            <div class="stat-label">TOOLS APPROVED</div>
            <div class="stat-value">{st.session_state.approved_count}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with stat_col3:
    st.markdown(
        f"""
        <div class="stat-card stat-magenta" style="animation-delay:0.2s, 0.4s;">
            <div class="stat-label">TOOLS DENIED</div>
            <div class="stat-value">{st.session_state.denied_count}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with stat_col4:
    st.markdown(
        f"""
        <div class="stat-card stat-yellow" style="animation-delay:0.3s, 0.6s;">
            <div class="stat-label">SESSION ID</div>
            <div class="stat-value">{st.session_state.thread_id[:8]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with stat_col5:
    st.markdown(
        f"""
        <div class="stat-card stat-white" style="animation-delay:0.4s, 0.8s;">
            <div class="stat-label">MODE</div>
            <div class="stat-value">{st.session_state.mode.upper()}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# CHAT HISTORY
# ============================================================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ============================================================
# TOOL APPROVAL UI
# ============================================================

if st.session_state.pending_approval:
    info = st.session_state.pending_approval

    arg_rows = "".join(
        f'<div class="perm-row"><span class="perm-key">{str(key).upper()}</span>'
        f'<span class="perm-val">{value}</span></div>'
        for key, value in info["tool_args"].items()
    )

    with st.chat_message("assistant"):
        st.markdown(
            f"""
            <div class="approval-box">
                <div class="approval-title">🔐 PERMISSION REQUEST</div>
                <div class="perm-row">
                    <span class="perm-key">TOOL</span>
                    <span class="perm-val">{info['tool_name']}</span>
                </div>
                {arg_rows}
                <div class="approval-status">&gt; AWAITING HUMAN APPROVAL...</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("✅ APPROVE", use_container_width=True, type="primary"):
                st.session_state.approved_count += 1
                with st.spinner("Running tool..."):
                    run_agent(Command(resume="yes"))
                st.rerun()

        with col2:
            if st.button("❌ DENY", use_container_width=True):
                st.session_state.denied_count += 1
                with st.spinner("Skipping tool..."):
                    run_agent(Command(resume="no"))
                st.rerun()

else:
    user_input = st.chat_input("Ask about weather or news in a Pakistani city...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Thinking..."):
            run_agent({"messages": [{"role": "user", "content": user_input}]})

        st.rerun()


# ============================================================
# AUTO-SCROLL TO LATEST CONTENT
# ============================================================

st.markdown('<div id="scroll-anchor"></div>', unsafe_allow_html=True)

components.html(
    """
    <script>
    (function () {
        try {
            var doc = window.parent.document;

            function getScrollContainer() {
                var selectors = [
                    'section[data-testid="stMain"]',
                    'div[data-testid="stAppViewContainer"]',
                    'section.main',
                    '.main'
                ];
                for (var i = 0; i < selectors.length; i++) {
                    var el = doc.querySelector(selectors[i]);
                    if (el && el.scrollHeight > el.clientHeight) {
                        return el;
                    }
                }
                return doc.scrollingElement || doc.documentElement;
            }

            function scrollToBottom() {
                var el = getScrollContainer();
                if (el) {
                    el.scrollTop = el.scrollHeight;
                }
            }

            // Poll and force the scroll position directly (no animation),
            // so there's nothing that can stack, conflict, or snap back.
            // This runs for ~3s after each rerun, which comfortably covers
            // spinners, tool calls, and streamed responses finishing.
            var attempts = 0;
            var maxAttempts = 20;
            var interval = setInterval(function () {
                scrollToBottom();
                attempts += 1;
                if (attempts >= maxAttempts) {
                    clearInterval(interval);
                }
            }, 150);
        } catch (e) {}
    })();
    </script>
    """,
    height=0,
)