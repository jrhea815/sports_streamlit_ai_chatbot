import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# --- Env / API key ---
load_dotenv()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# --- Streamlit basics ---
st.set_page_config(page_title="Sports CSV Chat", page_icon="🏈", layout="centered")
st.title("🏈 Sports CSV Chat")

CSV_PATH_DEFAULT = "stats.csv"   # <-- your repo file

@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def make_agent(df: pd.DataFrame):
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        agent_type="openai-tools",     # tool-calling; stable parsing
        verbose=False,
        allow_dangerous_code=True,     # lets the agent execute pandas code
        handle_parsing_errors=True,
    )
    return agent

# --- Sidebar: controls ---
st.sidebar.subheader("Data source")
csv_path = st.sidebar.text_input("Path to CSV", CSV_PATH_DEFAULT)
reload_btn = st.sidebar.button("Load / Reload CSV", type="primary")
st.sidebar.caption("Default is ./stats.csv in this repo.")

# --- Load DF & agent on first run or when reloaded ---
if "df" not in st.session_state or reload_btn:
    try:
        st.session_state.df = load_csv(csv_path)
        st.session_state.agent = make_agent(st.session_state.df)
        st.sidebar.success(f"Loaded {len(st.session_state.df):,} rows ✅")
    except Exception as e:
        st.sidebar.error(f"Could not load CSV at '{csv_path}': {e}")

# Preview the data
if "df" in st.session_state:
    with st.expander("Preview stats.csv (top 10)", expanded=False):
        st.dataframe(st.session_state.df.head(10), use_container_width=True)
    with st.expander("Columns", expanded=False):
        st.code(", ".join(st.session_state.df.columns.tolist()))

# Reset chat
colA, colB = st.columns(2)
with colA:
    if st.button("Reset chat"):
        st.session_state.messages = []

st.caption("Ask things like: *'Amon-Ra St. Brown receiving yards last 5'*, *'Average rushing yards for Christian McCaffrey by week in 2025'*.")

# --- Chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --- Chat input ---
q = st.chat_input("Ask about player performance…")
if q:
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    if "agent" not in st.session_state:
        with st.chat_message("assistant"):
            st.warning("Load a CSV first from the sidebar.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                # Give the model a tiny nudge to be concise and show code results
                instruction = (
                    "Be concise. Prefer tables for results. "
                    "If you compute 'last N', show which rows you used."
                )
                out = st.session_state.agent.invoke({"input": f"{instruction}\n\n{q}"})
                ans = out.get("output", str(out))
                st.markdown(ans)

        st.session_state.messages.append({"role": "assistant", "content": ans})
