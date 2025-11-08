# =====================================================
# 🗣️ ADVANCED AI DEBATE ARENA (Enhanced Version)
# =====================================================
# ✅ Features:
# - Choose model, temperature, and debate length
# - Optional user input during debate
# - Automatic debate summary after rounds
# - Clearer code structure & inline notes
# =====================================================

import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv

# --- 1️⃣ LOAD CONFIGURATION & API KEY ---
load_dotenv()
try:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except:
    st.error("❌ OpenAI API Key not found. Please create a '.env' file with OPENAI_API_KEY='your-key'.")
    st.stop()

# --- 2️⃣ DEFINE AI ROLES ---
PRO_AGENT_PROMPT = (
    "You are the **Pro-Side** debater. Your mission is to convincingly argue FOR the given topic. "
    "Be persuasive, logical, and concise. Always address your opponent’s last argument directly."
)

CON_AGENT_PROMPT = (
    "You are the **Con-Side** debater. Your mission is to convincingly argue AGAINST the given topic. "
    "Be critical, analytical, and precise. Directly refute the last statement made by your opponent."
)

SUMMARY_PROMPT = (
    "You are a neutral moderator. Summarize the entire debate objectively, highlighting the strongest "
    "points made by both sides and identifying any consensus or remaining disagreements."
)

# --- 3️⃣ CORE AI CALL FUNCTION ---
def get_ai_response(role_prompt, history, model="gpt-4o-mini", temperature=0.7):
    """
    Generates a debate response using OpenAI's API.
    Parameters:
        - role_prompt: system message defining the debater's role
        - history: list of all messages exchanged
        - model: chosen model name
        - temperature: controls creativity level (0 = strict, 1 = creative)
    """
    messages = [{"role": "system", "content": role_prompt}]
    messages.extend(history)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=300,
    )
    return response.choices[0].message.content


# --- 4️⃣ SESSION STATE INITIALIZATION ---
def init_state():
    """Initializes Streamlit session state variables."""
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'topic' not in st.session_state:
        st.session_state.topic = ""
    if 'current_turn' not in st.session_state:
        st.session_state.current_turn = 'PRO'
    if 'turn_count' not in st.session_state:
        st.session_state.turn_count = 0
    if 'debate_ended' not in st.session_state:
        st.session_state.debate_ended = False


# --- 5️⃣ DEBATE TURN LOGIC ---
def debate_turn(model, temperature, max_turns):
    """Executes one turn of the debate and updates session state."""
    if st.session_state.debate_ended:
        return

    current_turn = st.session_state.current_turn
    if current_turn == 'PRO':
        role_prompt = PRO_AGENT_PROMPT
        agent_name = "🤖 Pro-Side"
        next_turn = 'CON'
    else:
        role_prompt = CON_AGENT_PROMPT
        agent_name = "🧠 Con-Side"
        next_turn = 'PRO'

    with st.spinner(f"{agent_name} is formulating a rebuttal..."):
        if not st.session_state.history:
            # First turn → Opening statement
            first_prompt = f"The debate topic is: '{st.session_state.topic}'. Please give your opening statement."
            reply = get_ai_response(role_prompt, [{"role": "user", "content": first_prompt}], model, temperature)
        else:
            reply = get_ai_response(role_prompt, st.session_state.history, model, temperature)

    # Save response and switch turn
    st.session_state.history.append({"role": "assistant", "name": agent_name, "content": reply})
    st.session_state.current_turn = next_turn
    st.session_state.turn_count += 1

    # Check if debate should end
    if st.session_state.turn_count >= max_turns:
        st.session_state.debate_ended = True


# --- 6️⃣ DEBATE SUMMARY FUNCTION ---
def summarize_debate(model="gpt-4o-mini"):
    """Creates a neutral summary of the debate using AI."""
    history_text = "\n".join(
        [f"{msg['name']}: {msg['content']}" for msg in st.session_state.history]
    )
    summary_prompt = [{"role": "system", "content": SUMMARY_PROMPT},
                      {"role": "user", "content": history_text}]
    summary = client.chat.completions.create(
        model=model,
        messages=summary_prompt,
        temperature=0.5,
        max_tokens=300
    )
    return summary.choices[0].message.content


# --- 7️⃣ STREAMLIT UI ---
st.set_page_config(page_title="🤖 Advanced AI Debate Arena", layout="wide")
st.title("🗣️ Advanced AI Debate Arena")
st.caption("Watch two AI agents argue like professionals — and even join the discussion yourself!")

init_state()

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("⚙️ Debate Settings")
    topic_input = st.text_input("Debate Topic:", placeholder="e.g., 'AI will replace most jobs'")
    model_choice = st.selectbox("Choose Model:", ["gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"])
    creativity = st.slider("Creativity (temperature)", 0.0, 1.0, 0.7)
    max_turns = st.slider("Number of Turns", 2, 10, 6, step=2)
    st.divider()
    user_input = st.text_area("💬 Optional: Add your own argument", placeholder="Type your own rebuttal...")

    col1, col2 = st.columns(2)
    with col1:
        start = st.button("Start / Continue ▶️", use_container_width=True)
    with col2:
        if st.button("Reset Debate 🔄", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# --- MAIN DISPLAY AREA ---
if topic_input and topic_input != st.session_state.topic:
    st.session_state.topic = topic_input
    st.session_state.history = []
    st.session_state.current_turn = 'PRO'
    st.session_state.turn_count = 0
    st.session_state.debate_ended = False

if st.session_state.topic:
    st.subheader(f"🎯 Topic: {st.session_state.topic}")
    st.markdown("---")
else:
    st.info("Enter a topic in the sidebar and press ▶️ to start the debate!")

# Display debate messages
for msg in st.session_state.history:
    with st.chat_message(msg['name'].split()[1].lower()):
        st.markdown(f"**{msg['name']}**: {msg['content']}")

# Allow user argument insertion
if user_input and start:
    st.session_state.history.append({"role": "user", "name": "🧍User", "content": user_in
