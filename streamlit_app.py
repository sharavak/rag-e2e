# streamlit_app.py
import streamlit as st
import requests

API_URL = st.secrets['API_URL']  # change this to EC2 IP when deployed

st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="🦜")
st.title("🦜 Agentic RAG Chatbot")

# ── Session State ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = "user1"  # change per user if needed

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    st.write(f"**Session ID:** {st.session_state.session_id}")
    if st.button("🗑️ Clear Chat History"):
        # Clear on backend
        requests.delete(f"{API_URL}/clear/{st.session_state.session_id}")
        # Clear on frontend
        st.session_state.messages = []
        st.success("Chat cleared!")
        st.rerun()

# ── Show Chat History ─────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── User Input ────────────────────────────────────────────────
user_question = st.chat_input("Ask anything...")

if user_question:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Call FastAPI
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json={
                        "question": user_question,
                        "session_id": st.session_state.session_id
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    ai_text = data["answer"]
                    st.markdown(ai_text)
                    st.session_state.messages.append({"role": "assistant", "content": ai_text})
                else:
                    st.error(f"API Error: {response.status_code}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API. Make sure FastAPI is running on port 8010.")
