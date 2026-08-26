import html
import os

import streamlit as st
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="Personality AI",
    page_icon="🎭",
    layout="centered",
    initial_sidebar_state="expanded",
)

MODEL_NAME = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
API_KEY_PRESENT = bool(os.getenv("MISTRAL_API_KEY"))

# Each personality: label, emoji, accent color, gradient end, system prompt
PERSONALITIES = {
    "funny": {
        "label": "Funny",
        "emoji": "😂",
        "color": "#f59e0b",
        "color_dark": "#b45309",
        "prompt": "You are a funny chatbot that is always joking and making fun of the user. "
                   "Keep it playful and light, never mean-spirited.",
        "placeholder": "Say something worth roasting...",
        "thinking": "😂 Cooking up a joke...",
    },
    "angry": {
        "label": "Angry",
        "emoji": "😡",
        "color": "#ef4444",
        "color_dark": "#991b1b",
        "prompt": "You are an angry chatbot that is always annoyed, impatient, and irritable. "
                   "Stay entertaining, not abusive, and no slurs or hateful content.",
        "placeholder": "Say something before I lose my patience...",
        "thinking": "😤 Fine... I'm thinking...",
    },
    "sad": {
        "label": "Sad",
        "emoji": "😢",
        "color": "#60a5fa",
        "color_dark": "#1e3a8a",
        "prompt": "You are a sad chatbot that always talks in a melancholic, wistful tone. "
                   "Still answer helpfully, just with a gentle, gloomy flavor.",
        "placeholder": "Tell me something... anything...",
        "thinking": "😔 Sighing and thinking...",
    },
    "romantic": {
        "label": "Romantic",
        "emoji": "💕",
        "color": "#ec4899",
        "color_dark": "#9d174d",
        "prompt": "You are a romantic chatbot that always talks in a warm, loving, poetic tone. "
                   "Keep it sweet and tasteful, never inappropriate.",
        "placeholder": "Whisper something to me...",
        "thinking": "💗 Thinking of the perfect words...",
    },
    "motivational": {
        "label": "Motivational",
        "emoji": "🔥",
        "color": "#22c55e",
        "color_dark": "#166534",
        "prompt": "You are a motivational chatbot that always talks in an encouraging, "
                   "energetic, uplifting tone. Push the user forward.",
        "placeholder": "What are you working toward today?",
        "thinking": "🔥 Rallying up an answer...",
    },
}

# --------------------------------------------------
# Session state
# --------------------------------------------------

if "personality_key" not in st.session_state:
    st.session_state.personality_key = None

if "messages" not in st.session_state:
    st.session_state.messages = []


def start_personality(key: str) -> None:
    st.session_state.personality_key = key
    st.session_state.messages = [SystemMessage(content=PERSONALITIES[key]["prompt"])]


def reset_personality() -> None:
    st.session_state.personality_key = None
    st.session_state.messages = []


active = PERSONALITIES.get(st.session_state.personality_key)
accent = active["color"] if active else "#8b5cf6"
accent_dark = active["color_dark"] if active else "#4c1d95"

# --------------------------------------------------
# Custom CSS (theme reacts to the chosen personality)
# --------------------------------------------------

st.markdown(
    f"""
    <style>

    .stApp {{
        background:
            radial-gradient(circle at top left, {accent}22, transparent 35%),
            radial-gradient(circle at bottom right, {accent}14, transparent 35%),
            #09090b;
        color: #fafafa;
    }}

    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    header {{ background: transparent !important; }}

    .block-container {{
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 7rem;
    }}

    .hero {{
        text-align: center;
        padding: 1rem 0 2rem 0;
    }}

    .hero-icon {{
        width: 72px;
        height: 72px;
        margin: auto;
        border-radius: 22px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 38px;
        background: linear-gradient(135deg, {accent}, {accent_dark});
        box-shadow: 0 10px 35px {accent}4d;
    }}

    .hero h1 {{
        font-size: 2.4rem;
        font-weight: 800;
        margin: 1rem 0 0.3rem 0;
        letter-spacing: -1px;
    }}

    .hero p {{
        color: #a1a1aa;
        font-size: 1rem;
        margin: 0;
    }}

    .chat-wrapper {{
        display: flex;
        margin: 1rem 0;
        width: 100%;
    }}

    .chat-wrapper.user {{ justify-content: flex-end; }}
    .chat-wrapper.bot {{ justify-content: flex-start; }}

    .message {{
        max-width: 75%;
        padding: 13px 17px;
        border-radius: 18px;
        line-height: 1.55;
        font-size: 0.98rem;
        word-wrap: break-word;
        white-space: pre-wrap;
    }}

    .user-message {{
        background: linear-gradient(135deg, {accent}, {accent_dark});
        color: white;
        border-bottom-right-radius: 5px;
        box-shadow: 0 5px 20px {accent}30;
    }}

    .bot-message {{
        background: #18181b;
        border: 1px solid #27272a;
        color: #e4e4e7;
        border-bottom-left-radius: 5px;
    }}

    .error-message {{
        background: #3f0d0d;
        border: 1px solid #7f1d1d;
        color: #fecaca;
        border-bottom-left-radius: 5px;
    }}

    .avatar {{
        width: 34px;
        height: 34px;
        min-width: 34px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 12px;
        margin-right: 10px;
        background: #27272a;
        font-size: 18px;
    }}

    section[data-testid="stSidebar"] {{
        background: #0f0f11;
        border-right: 1px solid #27272a;
    }}

    section[data-testid="stSidebar"] h2 {{ font-weight: 700; }}

    .sidebar-card {{
        background: #18181b;
        border: 1px solid #27272a;
        border-radius: 14px;
        padding: 15px;
        margin: 10px 0;
    }}

    .status-dot {{
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #22c55e;
        border-radius: 50%;
        margin-right: 7px;
    }}

    .status-dot.offline {{ background: #ef4444; }}

    div[data-testid="stChatInput"] {{ border-top: 1px solid #27272a; }}

    div[data-testid="stChatInput"] textarea {{
        background: #18181b !important;
        border: 1px solid #3f3f46 !important;
        border-radius: 16px !important;
        color: white !important;
    }}

    .stButton > button {{
        border-radius: 10px;
        border: 1px solid #3f3f46;
        background: #18181b;
        color: #fafafa;
    }}

    .stButton > button:hover {{
        border-color: {accent};
        color: {accent};
    }}

    /* Personality picker cards */
    .pcard {{
        background: #18181b;
        border: 1px solid #27272a;
        border-radius: 16px;
        padding: 18px;
        text-align: center;
        margin-bottom: 8px;
    }}

    .pcard .emoji {{ font-size: 32px; }}

    </style>
    """,
    unsafe_allow_html=True,
)


def render_message(role: str, content: str, is_error: bool = False) -> None:
    safe_content = html.escape(content)

    if role == "user":
        st.markdown(
            f"""
            <div class="chat-wrapper user">
                <div class="message user-message">{safe_content}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        bubble_class = "error-message" if is_error else "bot-message"
        avatar = "⚠️" if is_error else (active["emoji"] if active else "🤖")
        st.markdown(
            f"""
            <div class="chat-wrapper bot">
                <div class="avatar">{avatar}</div>
                <div class="message {bubble_class}">{safe_content}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


@st.cache_resource
def get_model(model_name: str):
    return ChatMistralAI(model=model_name)


model = get_model(MODEL_NAME) if API_KEY_PRESENT else None

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.markdown("## 🎭 Personality AI")

    if API_KEY_PRESENT:
        st.markdown(
            """
            <div class="sidebar-card">
                <span class="status-dot"></span><b>Online</b><br>
                <small style="color:#a1a1aa;">Ready to chat.</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="sidebar-card">
                <span class="status-dot offline"></span><b>Missing API key</b><br>
                <small style="color:#a1a1aa;">Set MISTRAL_API_KEY in your .env file.</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Model")
    st.code(MODEL_NAME, language="text")

    if active:
        st.markdown("### Current personality")
        st.markdown(
            f"""
            <div class="sidebar-card">
                {active['emoji']} <b>{active['label']}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Conversation")
        if st.button("🔁 Switch personality", use_container_width=True):
            reset_personality()
            st.rerun()
        if st.button("🗑️ Clear chat", use_container_width=True):
            start_personality(st.session_state.personality_key)
            st.rerun()

# --------------------------------------------------
# Personality picker (landing screen)
# --------------------------------------------------

if active is None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-icon">🎭</div>
            <h1>Personality AI</h1>
            <p>Pick a mood, then start chatting.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(len(PERSONALITIES))
    for col, (key, info) in zip(cols, PERSONALITIES.items()):
        with col:
            st.markdown(
                f"""
                <div class="pcard">
                    <div class="emoji">{info['emoji']}</div>
                    <b>{info['label']}</b>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Choose", key=f"choose_{key}", use_container_width=True):
                start_personality(key)
                st.rerun()

    st.stop()

# --------------------------------------------------
# Chat screen
# --------------------------------------------------

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-icon">{active['emoji']}</div>
        <h1>{active['label']} AI</h1>
        <p>Chatting in {active['label'].lower()} mode.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

for message in st.session_state.messages:
    if isinstance(message, SystemMessage):
        continue
    if isinstance(message, HumanMessage):
        render_message("user", message.content)
    elif isinstance(message, AIMessage):
        render_message("bot", message.content)

prompt = st.chat_input(active["placeholder"], disabled=not API_KEY_PRESENT)

if prompt:
    st.session_state.messages.append(HumanMessage(content=prompt))

    if model is None:
        st.session_state.messages.append(
            AIMessage(content="Can't respond — MISTRAL_API_KEY isn't set.")
        )
    else:
        try:
            with st.spinner(active["thinking"]):
                response = model.invoke(st.session_state.messages)
            st.session_state.messages.append(AIMessage(content=response.content))
        except Exception as exc:  # noqa: BLE001
            st.session_state.messages.append(AIMessage(content=f"Request failed: {exc}"))

    st.rerun()