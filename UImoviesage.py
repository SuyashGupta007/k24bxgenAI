import html
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_mistralai import ChatMistralAI

# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="MovieSage",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="expanded",
)

MODEL_NAME = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
API_KEY_PRESENT = bool(os.getenv("MISTRAL_API_KEY"))

# Flat-file "database" — no SQLite, just a JSON array on disk.
DB_PATH = Path(os.getenv("MOVIESAGE_DB_PATH", "moviesage_db.json"))

ACCENT = "#f59e0b"
ACCENT_DARK = "#b45309"

# --------------------------------------------------
# Schema
# --------------------------------------------------


class Movie(BaseModel):
    title: str
    release_year: Optional[str] = None
    director: Optional[str] = None
    genre: List[str] = []
    main_cast: List[str] = []
    language: Optional[str] = None
    country: Optional[str] = None
    rating: Optional[str] = None
    runtime: Optional[str] = None
    plot_overview: str
    budget: Optional[str] = None
    box_office_collection: Optional[str] = None


parser = PydanticOutputParser(pydantic_object=Movie)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are MovieSage, an AI assistant specialized in analyzing movies.

Your task is to take raw, unstructured information about a movie and
transform it into structured JSON data.

Follow these rules:

1. Extract information only from the provided text.
2. Do not invent information.
3. If information is missing, use null or an empty list where appropriate.
4. Generate a clean and concise plot overview.
5. Return the result according to these formatting instructions:

{format_instructions}
""",
        ),
        (
            "human",
            """
Analyze the following raw movie information:

{movie_text}
""",
        ),
    ]
)


@st.cache_resource
def get_model(model_name: str):
    return ChatMistralAI(model=model_name)


model = get_model(MODEL_NAME) if API_KEY_PRESENT else None

# --------------------------------------------------
# Flat-file storage helpers
# --------------------------------------------------


def load_db() -> list:
    if not DB_PATH.exists():
        return []
    try:
        with DB_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def save_to_db(movie: Movie) -> None:
    records = load_db()
    entry = movie.model_dump()
    entry["_saved_at"] = datetime.now(timezone.utc).isoformat()

    # Replace an existing entry with the same title + year instead of duplicating.
    records = [
        r
        for r in records
        if not (
            r.get("title", "").strip().lower() == movie.title.strip().lower()
            and r.get("release_year") == movie.release_year
        )
    ]
    records.append(entry)

    with DB_PATH.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "result" not in st.session_state:
    st.session_state.result = None
if "raw_response" not in st.session_state:
    st.session_state.raw_response = None
if "error" not in st.session_state:
    st.session_state.error = None

# --------------------------------------------------
# Styling
# --------------------------------------------------

st.markdown(
    f"""
    <style>
    .stApp {{
        background:
            radial-gradient(circle at top left, {ACCENT}1f, transparent 35%),
            radial-gradient(circle at bottom right, {ACCENT}14, transparent 35%),
            #09090b;
        color: #fafafa;
    }}
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    header {{ background: transparent !important; }}

    .block-container {{
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }}

    .hero {{ text-align: center; padding: 1rem 0 2rem 0; }}

    .hero-icon {{
        width: 72px;
        height: 72px;
        margin: auto;
        border-radius: 22px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 38px;
        background: linear-gradient(135deg, {ACCENT}, {ACCENT_DARK});
        box-shadow: 0 10px 35px {ACCENT}4d;
    }}

    .hero h1 {{
        font-size: 2.2rem;
        font-weight: 800;
        margin: 1rem 0 0.3rem 0;
        letter-spacing: -1px;
    }}

    .hero p {{ color: #a1a1aa; font-size: 1rem; margin: 0; }}

    .movie-card {{
        background: #18181b;
        border: 1px solid #27272a;
        border-radius: 16px;
        padding: 20px 22px;
        margin: 14px 0;
    }}

    .movie-title {{
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 2px;
    }}

    .movie-sub {{ color: #a1a1aa; font-size: 0.9rem; margin-bottom: 14px; }}

    .tag {{
        display: inline-block;
        background: {ACCENT}22;
        color: {ACCENT};
        border: 1px solid {ACCENT}55;
        border-radius: 999px;
        padding: 2px 10px;
        font-size: 0.8rem;
        margin: 2px 4px 2px 0;
    }}

    .field-label {{
        color: #a1a1aa;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 10px;
    }}

    .field-value {{ font-size: 0.98rem; }}

    .status-dot {{
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #22c55e;
        border-radius: 50%;
        margin-right: 7px;
    }}
    .status-dot.offline {{ background: #ef4444; }}

    .sidebar-card {{
        background: #18181b;
        border: 1px solid #27272a;
        border-radius: 14px;
        padding: 12px 14px;
        margin: 8px 0;
    }}

    .stButton > button {{
        border-radius: 10px;
        border: 1px solid #3f3f46;
        background: #18181b;
        color: #fafafa;
    }}
    .stButton > button:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}

    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Sidebar — status + saved movies ("database")
# --------------------------------------------------

with st.sidebar:
    st.markdown("## 🎬 MovieSage")

    if API_KEY_PRESENT:
        st.markdown(
            """
            <div class="sidebar-card">
                <span class="status-dot"></span><b>Online</b><br>
                <small style="color:#a1a1aa;">Ready to analyze.</small>
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

    st.markdown("### Saved movies")
    records = load_db()

    if not records:
        st.caption("Nothing saved yet.")
    else:
        for r in reversed(records):
            year = f" ({r['release_year']})" if r.get("release_year") else ""
            st.markdown(
                f"""
                <div class="sidebar-card">
                    <b>{html.escape(r.get('title', 'Untitled'))}</b>{html.escape(year)}
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.download_button(
            "⬇️ Download database (JSON)",
            data=json.dumps(records, indent=2, ensure_ascii=False),
            file_name="moviesage_db.json",
            mime="application/json",
            use_container_width=True,
        )

        if st.button("🗑️ Clear database", use_container_width=True):
            DB_PATH.write_text("[]", encoding="utf-8")
            st.rerun()

# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-icon">🎬</div>
        <h1>MovieSage</h1>
        <p>Paste a paragraph about a movie — get clean, structured data back.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Input
# --------------------------------------------------

para = st.text_area(
    "Movie paragraph",
    height=180,
    placeholder="Paste a raw paragraph about the movie here...",
    label_visibility="collapsed",
)

analyze_clicked = st.button(
    "✨ Analyze movie",
    use_container_width=True,
    disabled=not API_KEY_PRESENT,
)

if analyze_clicked:
    st.session_state.result = None
    st.session_state.raw_response = None
    st.session_state.error = None

    if not para.strip():
        st.session_state.error = "Paste a paragraph about the movie first."
    else:
        try:
            with st.spinner("Reading between the lines..."):
                final_prompt = prompt.invoke(
                    {
                        "movie_text": para,
                        "format_instructions": parser.get_format_instructions(),
                    }
                )
                res = model.invoke(final_prompt)

            try:
                movie = parser.parse(res.content)
                st.session_state.result = movie
                save_to_db(movie)
            except OutputParserException:
                st.session_state.raw_response = res.content
                st.session_state.error = (
                    "The model's reply couldn't be parsed into the expected "
                    "structure. Raw response is shown below."
                )
        except Exception as exc:  # noqa: BLE001
            st.session_state.error = f"Request failed: {exc}"

# --------------------------------------------------
# Output
# --------------------------------------------------

if st.session_state.error:
    st.error(st.session_state.error)
    if st.session_state.raw_response:
        st.code(st.session_state.raw_response, language="text")

if st.session_state.result:
    movie: Movie = st.session_state.result

    genre_tags = "".join(f'<span class="tag">{html.escape(g)}</span>' for g in movie.genre)
    cast = ", ".join(movie.main_cast) if movie.main_cast else "—"

    meta_bits = [
        b
        for b in [movie.release_year, movie.language, movie.country, movie.runtime]
        if b
    ]
    sub_line = " · ".join(meta_bits)

    st.markdown(
        f"""
        <div class="movie-card">
            <div class="movie-title">{html.escape(movie.title)}</div>
            <div class="movie-sub">{html.escape(sub_line)}</div>

            {genre_tags}

            <div class="field-label">Director</div>
            <div class="field-value">{html.escape(movie.director or '—')}</div>

            <div class="field-label">Main cast</div>
            <div class="field-value">{html.escape(cast)}</div>

            <div class="field-label">Plot overview</div>
            <div class="field-value">{html.escape(movie.plot_overview)}</div>

            <div class="field-label">Rating</div>
            <div class="field-value">{html.escape(movie.rating or '—')}</div>

            <div class="field-label">Budget</div>
            <div class="field-value">{html.escape(movie.budget or '—')}</div>

            <div class="field-label">Box office collection</div>
            <div class="field-value">{html.escape(movie.box_office_collection or '—')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("View raw JSON"):
        st.code(movie.model_dump_json(indent=2), language="json")

    st.caption(f"Saved to `{DB_PATH.name}`.")