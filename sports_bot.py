import os
from pathlib import Path
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import yaml

from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent


with open('secrets.yaml', "r") as f:
    secrets_data = yaml.safe_load(f)

openai_api = secrets_data['api_key']["key"]

# --- Streamlit basics ---
st.set_page_config(page_title="NFL Chatbot", page_icon="🏈", layout="wide")
st.title("🏈 NFL Chatbot")

CSV_PATH = Path(__file__).parent / "stats.csv"  # <- fixed file in your repo

@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"stats.csv not found at: {path}")
    return pd.read_csv(path)

def make_agent(df: pd.DataFrame):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key= openai_api)
    return create_pandas_dataframe_agent(
        llm,
        df,
        agent_type="openai-tools",
        verbose=False,
        allow_dangerous_code=True,   # lets the agent run pandas code
        handle_parsing_errors=True,
    )

# Load data + agent once
try:
    df = load_csv(CSV_PATH)
    agent = make_agent(df)
    st.success(f"Loaded stats.csv — {len(df):,} rows")
except Exception as e:
    st.error(f"Could not load stats.csv: {e}")
    st.stop()

# Preview
with st.expander("Preview stats.csv (top 10)", expanded=False):
    st.dataframe(df.head(10), use_container_width=True)
with st.expander("Columns", expanded=False):
    st.code(", ".join(df.columns.tolist()))

# Reset chat
if "messages" not in st.session_state:
    st.session_state.messages = []
if st.button("Reset chat"):
    st.session_state.messages = []

st.caption("Try: *'J. Gibbs rushing yards by game in 2025'*.")

# History
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat input
q = st.chat_input("Ask about player performance…")
if q:
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            instruction = (
                "Be concise. Prefer tables for results. "
                "If you compute 'last N', show which rows you used."
            )
            out = agent.invoke({"input": f"{instruction}\n\n{q}"})
            ans = out.get("output", str(out))
            st.markdown(ans)

    st.session_state.messages.append({"role": "assistant", "content": ans})
