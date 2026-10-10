"""
frontend/app.py

Streamlit chat UI for RAVEN. Talks to the GraphQL backend (src/api/main.py)
over HTTP - this is a separate process, not a direct Python import, so the
API server must be running first.

Run:
    1. In one terminal:  uvicorn src.api.main:app --reload
    2. In another terminal:  streamlit run frontend/app.py

The backend address can be changed with the RAVEN_API_URL environment
variable (Docker uses this to point at the backend container).
"""

import os

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_URL = os.getenv("RAVEN_API_URL", "http://localhost:8000/graphql")
REQUEST_TIMEOUT_SECONDS = 60  # LLM generation can take a little while

CHAT_QUERY = """
query Chat($question: String!, $topK: Int!) {
  chat(question: $question, topK: $topK) {
    answer
    cached
    latencyMs
    sources {
      title
      url
    }
  }
}
"""

STATUS_QUERY = """
query Status {
  collectionInfo {
    collection
    chunkCount
  }
  cacheInfo {
    backend
    hits
    misses
    entries
  }
}
"""


# ---------------------------------------------------------------------------
# Backend calls
# ---------------------------------------------------------------------------

def call_graphql(query: str, variables: dict = None) -> dict:
    """POST a GraphQL query to the backend. Raises on network/HTTP failure."""
    response = requests.post(
        API_URL,
        json={"query": query, "variables": variables or {}},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()

    if "errors" in payload and payload["errors"]:
        # GraphQL returns 200 even on resolver errors, with an "errors" list
        raise RuntimeError(payload["errors"][0].get("message", "Unknown GraphQL error"))

    return payload["data"]


def ask_question(question: str, top_k: int = 4) -> dict:
    data = call_graphql(CHAT_QUERY, {"question": question, "topK": top_k})
    return data["chat"]


def get_status() -> dict:
    """Knowledge-base size and cache stats, fetched in one request."""
    return call_graphql(STATUS_QUERY)


def format_meta(cached: bool, latency_ms: int) -> str:
    """Small caption shown under each answer."""
    took = f"{latency_ms} ms" if latency_ms < 1000 else f"{latency_ms / 1000:.1f} s"
    return f"⚡ from cache · {took}" if cached else f"🐦‍⬛ generated · {took}"


def render_sources(sources: list):
    if sources:
        with st.expander("Sources"):
            for s in sources:
                st.markdown(f"- [{s['title']}]({s['url']})")


def render_status(box):
    """Backend connection, knowledge-base size, and cache stats."""
    with box.container():
        try:
            status = get_status()
            info, cache_info = status["collectionInfo"], status["cacheInfo"]
            st.success("Backend connected")
            st.metric("Indexed chunks", info["chunkCount"])
            st.caption(f"Collection: {info['collection']}")

            st.subheader("Answer cache")
            st.caption(f"Backend: {cache_info['backend']}")
            col1, col2 = st.columns(2)
            col1.metric("Hits", cache_info["hits"])
            col2.metric("Misses", cache_info["misses"])
        except requests.exceptions.ConnectionError:
            st.error("Can't reach the backend.")
            st.caption("Make sure this is running in another terminal:\n\n`uvicorn src.api.main:app --reload`")
        except Exception as e:
            st.warning(f"Backend reachable, but something's off: {e}")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="RAVEN", page_icon="🐦‍⬛", layout="centered")

st.title("🐦‍⬛ RAVEN")
st.caption("Ask a question about skincare, grounded in real sources.")

# Sidebar: the status panel is a placeholder, filled in at the very end of
# the script - after any new question has been answered - so the cache
# hit/miss counters are never one question behind.
with st.sidebar:
    st.header("Knowledge base")
    status_box = st.empty()

    st.divider()
    top_k = st.slider("Chunks to retrieve per question", min_value=1, max_value=10, value=4)

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# Chat history lives in session_state so it survives reruns within one session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_sources(msg.get("sources"))
        if msg.get("meta"):
            st.caption(msg["meta"])

# New message input
question = st.chat_input("Ask a skincare question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        meta = None
        with st.spinner("Thinking..."):
            try:
                result = ask_question(question, top_k=top_k)
                answer = result["answer"]
                sources = result.get("sources", [])
                meta = format_meta(result["cached"], result["latencyMs"])
            except requests.exceptions.ConnectionError:
                answer = (
                    "I can't reach the backend right now. Make sure the API server "
                    "is running (`uvicorn src.api.main:app --reload`) and try again."
                )
                sources = []
            except Exception as e:
                answer = f"Something went wrong: {e}"
                sources = []

        st.markdown(answer)
        render_sources(sources)
        if meta:
            st.caption(meta)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources, "meta": meta}
    )

# Rendered last so the numbers include the question answered just above
render_status(status_box)
