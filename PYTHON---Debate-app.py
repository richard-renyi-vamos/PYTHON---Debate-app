import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv

# --- Configuration ---
# 1. Load environment variables (API Key)
# Make sure you have a .env file with OPENAI_API_KEY="your-api-key"
load_dotenv()
try:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except:
    st.error("OpenAI API Key not found. Please create a '.env' file or set the OPENAI_API_KEY environment variable.")
    st.stop()


# --- Agent Definitions ---
# System messages define the roles and debate rules for the AI
PRO_AGENT_PROMPT = (
    "You are the **Pro-Side** debater. Your goal is to strongly advocate "
    "FOR the given debate topic. Your tone is persuasive and professional. "
    "Keep your response concise and directly address the opponent's last point."
)

CON_AGENT_PROMPT = (
    "You are the **Con-Side** debater. Your goal is to strongly argue "
    "AGAINST the given debate topic. Your tone is analytical and challenging. "
    "Keep your response concise and directly refute the opponent's last point."
)

# --- Core LLM Logic ---
def get_debate_response(role_prompt, history):
    """Generates a response from an AI agent based on its role and the conversation history."""
    
    # The 'history' list contains all messages so far, which serves as context.
    messages = [
        {"role": "system", "content": role_prompt},
    ]
    messages.extend(history)
    
    # Call the OpenAI Chat Completion API
    response = client.chat.completions.create(
        model="gpt-3.5-turbo", # or gpt-4o-mini for better quality
        messages=messages,
        temperature=0.7,
        max_tokens=200,
    )
    return response.choices[0].message.content

# --- Streamlit App Initialization ---
def initialize_session_state():
    """Initializes the session state variables for the debate."""
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'topic' not in st.session_state:
        st.session_state['topic'] = ""
    if 'current_turn' not in st.session_state:
        st.session_state['current_turn'] = 'PRO' # PRO starts first

def run_debate_turn():
    """Executes one turn of the debate."""
    current_turn = st.session_state['current_turn']
    
    if current_turn == 'PRO':
        role_prompt = PRO_AGENT_PROMPT
        agent_name = "🤖 Pro-Side"
        next_turn = 'CON'
    else: # CON's turn
        role_prompt = CON_AGENT_PROMPT
        agent_name = "🧠 Con-Side"
        next_turn = 'PRO'
        
    with st.spinner(f"{agent_name} is formulating a rebuttal..."):
        # The PROMPT includes the debate topic to ensure initial grounding
        if not st.session_state['history']:
            # First turn: State the side on the topic
            full_prompt = (
                f"The debate topic is: '{st.session_state['topic']}'. "
                f"As the {agent_name.replace('🤖 ', '').replace('🧠 ', '')}, "
                "deliver your opening statement."
            )
            # Use only the system message for the first turn's context
            response_text = get_debate_response(role_prompt, 
                                                [{"role": "user", "content": full_prompt}])
        else:
            # Subsequent turns: Continue the conversation
            response_text = get_debate_response(role_prompt, st.session_state['history'])

    # Add the agent's message to the history and display it
    st.session_state['history'].append({"role": "assistant", "content": response_text, "name": agent_name})
    st.session_state['current_turn'] = next_turn # Switch turns

# --- Main App Structure ---

st.set_page_config(page_title="Python AI Debate App", layout="wide")
st.title("🗣️ The Python AI Debate Arena")
st.caption("A Streamlit app to watch two AI agents debate a topic.")

initialize_session_state()

# Sidebar for controls
with st.sidebar:
    st.header("Debate Setup")
    
    new_topic = st.text_input(
        "Enter a Debate Topic:",
        placeholder="e.g., 'The four-day work week should be standard'",
        key="topic_input"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("Start/Continue Debate ▶️", use_container_width=True)
    with col2:
        if st.button("Reset Debate 🔄", use_container_width=True):
            st.session_state['history'] = []
            st.session_state['topic'] = ""
            st.session_state['current_turn'] = 'PRO'
            st.rerun()

# --- Display Logic ---

if new_topic and new_topic != st.session_state['topic']:
    st.session_state['topic'] = new_topic
    st.session_state['history'] = [] # Reset history when topic changes
    st.session_state['current_turn'] = 'PRO'

if st.session_state['topic']:
    st.subheader(f"Topic: {st.session_state['topic']}")
    st.markdown("---")
else:
    st.info("Enter a topic in the sidebar and click 'Start Debate' to begin.")
    
# Display chat history
chat_placeholder = st.empty()
with chat_placeholder.container():
    for message in st.session_state['history']:
        # Use Streamlit's built-in chat elements
        with st.chat_message(message["name"].split()[1].lower()):
            st.write(message["content"])

# Run the next turn if the button is pressed
if start_button and st.session_state['topic']:
    run_debate_turn()
    st.rerun()
