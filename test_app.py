import os
import io
import base64
import tempfile
import time
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from groq import Groq
from gtts import gTTS
from flask_bcrypt import Bcrypt
from agent import get_conversation_response

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fluency Coach — Test", page_icon="🧪")

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY", "")
DATABASE_URL = st.secrets.get("DATABASE_URL") or os.environ.get("DATABASE_URL", "sqlite:///dev.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

MAX_TURNS = 5

TOPICS = [
    {"name": "Monday morning small talk",  "opening": "Hey, how was your weekend? Do anything interesting?"},
    {"name": "Talking about food",         "opening": "I have been trying to cook more at home. Do you enjoy cooking?"},
    {"name": "Talking about a hobby",      "opening": "I have been trying to get into running lately. Do you have any hobbies you are really into?"},
    {"name": "Talking about work",         "opening": "Work has been so busy for me lately. How about you, are things busy at your end?"},
    {"name": "Talking about a TV show",    "opening": "I just finished watching a really good series. Are you watching anything good right now?"},
    {"name": "Talking about pets",         "opening": "My neighbour just got a puppy and it is so cute! Do you have any pets?"},
    {"name": "At a coffee shop",           "opening": "I just tried that new cafe on the corner. Have you been there yet?"},
    {"name": "Talking about the weather",  "opening": "Can you believe how hot it has been lately? Is it like this where you live?"},
    {"name": "Recommending a restaurant",  "opening": "I had the most amazing dinner last night. Do you have a favourite restaurant around here?"},
    {"name": "Weekend plans",              "opening": "So what are you up to this weekend? Got anything fun planned?"},
    {"name": "Shopping",                   "opening": "I went to the mall yesterday and it was packed! Do you enjoy shopping?"},
    {"name": "Health and exercise",        "opening": "I have been trying to go to the gym more regularly. Do you exercise much?"},
    {"name": "Catching up with a friend",  "opening": "It feels like we have not talked in ages! What have you been up to lately?"},
    {"name": "Planning a trip",            "opening": "I am thinking about taking a trip somewhere next month. Have you travelled anywhere nice lately?"},
    {"name": "Talking about a movie",      "opening": "I watched a really good movie last night. Have you seen anything good recently?"},
]

# ── DB helpers (read-only) ────────────────────────────────────────────────────
@st.cache_resource
def get_engine():
    url = DATABASE_URL
    if "?" not in url:
        url = url + "?sslmode=require"
    return create_engine(url, pool_pre_ping=False)

def get_user(email: str):
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT id, name, email, password_hash, role FROM users WHERE email = :email"),
                {"email": email}
            ).fetchone()
        return result
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None

def get_student_level(user_id: int) -> int:
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT sa.overall_score
                FROM session_analysis sa
                JOIN sessions s ON sa.session_id = s.id
                WHERE s.user_id = :user_id
                ORDER BY sa.id DESC
                LIMIT 1
            """),
            {"user_id": user_id}
        ).fetchone()
    if result and result[0]:
        return int(round(result[0]))
    return 5  # default mid-level

def check_password(stored_hash: str, password: str) -> bool:
    from flask import Flask
    _app = Flask(__name__)
    _bcrypt = Bcrypt(_app)
    with _app.app_context():
        return _bcrypt.check_password_hash(stored_hash, password)

# ── Audio helpers ─────────────────────────────────────────────────────────────
groq_client = Groq(api_key=GROQ_API_KEY)

def make_audio_b64(text: str) -> str:
    if not text:
        return ""
    text = text[:500]
    for attempt in range(3):
        try:
            tts = gTTS(text=text, lang="en", slow=False)
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            buf.seek(0)
            return base64.b64encode(buf.read()).decode()
        except Exception:
            if attempt < 2:
                time.sleep(2)
    return ""

def transcribe_audio(audio_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        f.flush()
        with open(f.name, "rb") as af:
            result = groq_client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=af,
                language="en",
            )
    return result.text.strip()

def autoplay_audio(b64: str):
    if b64:
        st.markdown(
            f'<audio autoplay style="display:none">'
            f'<source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>',
            unsafe_allow_html=True,
        )

# ── Session state init ────────────────────────────────────────────────────────
if "logged_in"      not in st.session_state: st.session_state.logged_in      = False
if "user_id"        not in st.session_state: st.session_state.user_id        = None
if "user_name"      not in st.session_state: st.session_state.user_name      = ""
if "student_level"  not in st.session_state: st.session_state.student_level  = 5
if "topic_index"    not in st.session_state: st.session_state.topic_index    = 0
if "messages"       not in st.session_state: st.session_state.messages       = []
if "history"        not in st.session_state: st.session_state.history        = []
if "turn_count"     not in st.session_state: st.session_state.turn_count     = 0
if "pending_audio"  not in st.session_state: st.session_state.pending_audio  = None
if "conv_started"   not in st.session_state: st.session_state.conv_started   = False
if "conv_finished"  not in st.session_state: st.session_state.conv_finished  = False

# ── Login screen ──────────────────────────────────────────────────────────────
if not st.session_state.logged_in:
    st.title("🧪 Fluency Coach — Test App")
    st.caption("Log in with your existing account to test the conversation agent.")
    st.divider()

    with st.form("login_form"):
        email    = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")

    if submitted:
        user = get_user(email.strip().lower())
        if user and check_password(user.password_hash, password):
            st.session_state.logged_in     = True
            st.session_state.user_id       = user.id
            st.session_state.user_name     = user.name
            st.session_state.student_level = get_student_level(user.id)
            st.rerun()
        else:
            st.error("Invalid email or password.")
    st.stop()

# ── Logged in — main app ──────────────────────────────────────────────────────
current_topic = TOPICS[st.session_state.topic_index % len(TOPICS)]

# Header
col1, col2 = st.columns([4, 1])
with col1:
    st.title("🧪 Fluency Coach — Test App")
    st.caption(
        f"Logged in as **{st.session_state.user_name}** · "
        f"Level **{st.session_state.student_level}** · "
        f"Topic **{st.session_state.topic_index + 1}** of **{len(TOPICS)}**"
    )
with col2:
    if st.button("Log out"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

st.divider()

# Topic label
st.markdown(f"**Topic:** {current_topic['name']}")
st.progress((st.session_state.turn_count) / MAX_TURNS)
st.caption(f"Turn {min(st.session_state.turn_count + 1, MAX_TURNS)} of {MAX_TURNS}")
st.markdown("---")

# Start conversation — add opening message once
if not st.session_state.conv_started:
    opening     = current_topic["opening"]
    audio_b64   = make_audio_b64(opening)
    st.session_state.messages.append({"role": "assistant", "content": opening})
    st.session_state.pending_audio = audio_b64
    st.session_state.conv_started  = True

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Play pending audio
if st.session_state.pending_audio:
    autoplay_audio(st.session_state.pending_audio)
    st.session_state.pending_audio = None

# ── Conversation finished — show New Conversation button ──────────────────────
if st.session_state.conv_finished:
    st.markdown("---")
    next_index = (st.session_state.topic_index + 1) % len(TOPICS)
    next_topic = TOPICS[next_index]
    btn_label  = (
        f"New conversation → {next_topic['name']}"
        if next_index != 0
        else f"🔄 All topics done! Start over → {next_topic['name']}"
    )
    if st.button(btn_label, type="primary"):
        st.session_state.topic_index   = next_index
        st.session_state.messages      = []
        st.session_state.history       = []
        st.session_state.turn_count    = 0
        st.session_state.pending_audio = None
        st.session_state.conv_started  = False
        st.session_state.conv_finished = False
        st.rerun()
    st.stop()

# ── Input ─────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 3])
with col1:
    st.markdown("**🎤 Tap to start · tap again to stop**")
    from audio_recorder_streamlit import audio_recorder
    audio_bytes = audio_recorder(
        text="",
        recording_color="#e85d04",
        neutral_color="#6c757d",
        icon_size="2x",
        pause_threshold=3.0,
        key=f"recorder_{st.session_state.turn_count}_{st.session_state.topic_index}",
    )
with col2:
    st.markdown("**⌨️ Or type your answer:**")
    text_input = st.chat_input("Type here...")

# ── Process answer ────────────────────────────────────────────────────────────
def handle_answer(user_text: str):
    last_question = (
        st.session_state.messages[-1]["content"]
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant"
        else current_topic["opening"]
    )

    st.session_state.messages.append({"role": "user", "content": user_text})
    st.session_state.turn_count += 1
    is_last = st.session_state.turn_count >= MAX_TURNS

    with st.spinner("Thinking..."):
        comment, suggestion, question = get_conversation_response(
            user_text,
            st.session_state.history,
            current_topic["name"],
            st.session_state.student_level,
        )

    # Update history (read-only — not saved to DB)
    st.session_state.history.append({"role": "student",   "content": user_text})
    st.session_state.history.append({"role": "assistant", "content": comment})

    # Show comment
    st.session_state.messages.append({"role": "assistant", "content": f"💬 {comment}"})

    if is_last:
        closing = "Great conversation! Click 'New conversation' to try the next topic."
        st.session_state.messages.append({"role": "assistant", "content": closing})
        st.session_state.pending_audio  = make_audio_b64(closing)
        st.session_state.conv_finished  = True
    else:
        st.session_state.messages.append({"role": "assistant", "content": question})
        st.session_state.pending_audio = make_audio_b64(question)

    st.rerun()

if audio_bytes and len(audio_bytes) > 1000:
    with st.spinner("Transcribing..."):
        try:
            user_text = transcribe_audio(audio_bytes)
        except Exception as e:
            st.error(f"Transcription failed: {e}")
            user_text = None
    if user_text:
        handle_answer(user_text)

elif text_input:
    handle_answer(text_input)
