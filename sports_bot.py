import os
from pathlib import Path
import pandas as pd
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent


# --- Streamlit basics ---
st.set_page_config(page_title="NFL Query Chat", page_icon="🏈", layout="wide")
st.title("🏈 NFL Query Chat")

CSV_PATH = Path(__file__).parent / "stats.csv"  # fixed file in your repo


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"stats.csv not found at: {path}")
    return pd.read_csv(path)


def make_agent(df: pd.DataFrame, api_key: str):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    return create_pandas_dataframe_agent(
        llm,
        df,
        agent_type="openai-tools",
        verbose=False,
        allow_dangerous_code=True,
        handle_parsing_errors=True,
    )


def verify_api_key(api_key: str) -> tuple[bool, str]:
    """
    Verifies the key by making a minimal call.
    Returns (ok, message).
    """
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
        _ = llm.invoke("Reply with 'ok' only.")
        return True, "✅ API key verified."
    except Exception as e:
        return False, f"❌ API key verification failed: {e}"


# --------------------------
# Sidebar: API key + connect
# --------------------------
with st.sidebar:
    st.header("🔑 Connect OpenAI")

    if "api_verified" not in st.session_state:
        st.session_state.api_verified = False
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = ""

    api_key_input = st.text_input(
        "Paste your OpenAI API key",
        type="password",
        placeholder="sk-...",
        value=st.session_state.openai_api_key,
    )

    col1, col2 = st.columns(2)
    with col1:
        connect = st.button("Connect", use_container_width=True)
    with col2:
        disconnect = st.button("Disconnect", use_container_width=True)
    

    if disconnect:
        st.session_state.api_verified = False
        st.session_state.openai_api_key = ""
        st.toast("Disconnected.")
        st.rerun()

    if connect:
        st.session_state.openai_api_key = api_key_input.strip()
        if not st.session_state.openai_api_key:
            st.session_state.api_verified = False
            st.error("Please paste an API key first.")
        else:
            ok, msg = verify_api_key(st.session_state.openai_api_key)
            st.session_state.api_verified = ok
            if ok:
                st.success(msg)
            else:
                st.error(msg)

    if not st.session_state.api_verified:
        st.info("Paste your API key and click **Connect** to start.")
        st.stop()


# ----------------
# Load CSV + agent
# ----------------
try:
    df = load_csv(CSV_PATH)
    agent = make_agent(df, st.session_state.openai_api_key)
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
