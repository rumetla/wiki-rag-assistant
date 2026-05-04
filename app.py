"""Streamlit chat UI for the Local Wikipedia RAG Assistant.

Run with::

    streamlit run app.py

Features:
- Chat-style conversation with streamed assistant responses.
- Expander showing retrieved sources (entity, type, distance, Wikipedia URL).
- Sidebar with index status (people / places counts), model info, reset.
- Honors `.cursorrules`: 100% local, no external services touched.
"""

from __future__ import annotations

from typing import Iterator

import streamlit as st

from rag_engine import (
    LLM_MODEL,
    Source,
    answer_stream,
)
from vector_store import EMBED_MODEL, VectorStore


st.set_page_config(
    page_title="Local Wikipedia RAG",
    page_icon=":books:",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_store() -> VectorStore:
    """Cache one VectorStore instance per Streamlit process."""
    return VectorStore()


def _format_sources(sources: list[Source]) -> str:
    """Return a markdown summary of retrieved sources."""
    if not sources:
        return "_No sources retrieved._"
    lines: list[str] = []
    for i, s in enumerate(sources, start=1):
        snippet = s.text.strip().replace("\n", " ")
        if len(snippet) > 280:
            snippet = snippet[:280] + "..."
        url_part = f" ([wikipedia]({s.url}))" if s.url else ""
        lines.append(
            f"**[{i}] {s.name}** _(type: {s.type}, distance: {s.distance:.3f})_{url_part}\n\n"
            f"> {snippet}"
        )
    return "\n\n".join(lines)


def _stream_with_capture(token_iter: Iterator[str], buffer: list[str]) -> Iterator[str]:
    """Pass tokens through to Streamlit while capturing them for history."""
    for tok in token_iter:
        buffer.append(tok)
        yield tok


def _render_sidebar(store: VectorStore) -> None:
    with st.sidebar:
        st.header("Local Wikipedia RAG")
        st.caption("Fully local. No external APIs.")

        st.subheader("Index status")
        counts = store.count_by_type()
        total = counts["person"] + counts["place"]
        if total == 0:
            st.warning("Index is empty. Run `python ingest.py` first.")
        else:
            st.metric("People chunks", counts["person"])
            st.metric("Place chunks", counts["place"])

        st.subheader("Models (Ollama)")
        st.write(f"LLM: `{LLM_MODEL}`")
        st.write(f"Embeddings: `{EMBED_MODEL}`")

        st.divider()
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def _render_history() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("Retrieved sources", expanded=False):
                    st.markdown(_format_sources(msg["sources"]))


def main() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    store = get_store()
    _render_sidebar(store)

    st.title("Local Wikipedia RAG Assistant")
    st.caption(
        "Ask about famous people or places. The assistant answers strictly "
        "from local Wikipedia content; if it doesn't know, it says so."
    )

    _render_history()

    user_input = st.chat_input("Ask about a person or place...")
    if not user_input:
        return

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            token_iter, sources, route_type, grounded = answer_stream(store, user_input)
        except Exception as e:
            err = (
                f"**Error talking to Ollama:** {e}\n\n"
                "Make sure Ollama is running locally (`ollama serve`) and that "
                f"the models `{LLM_MODEL}` and `{EMBED_MODEL}` are pulled."
            )
            st.error(err)
            st.session_state.messages.append(
                {"role": "assistant", "content": err, "sources": []}
            )
            return

        st.caption(f"route: `{route_type}` | grounded: `{grounded}`")

        captured: list[str] = []
        st.write_stream(_stream_with_capture(token_iter, captured))
        full_answer = "".join(captured).strip() or "I don't know."

        if sources:
            with st.expander("Retrieved sources", expanded=False):
                st.markdown(_format_sources(sources))

        st.session_state.messages.append(
            {"role": "assistant", "content": full_answer, "sources": sources}
        )


if __name__ == "__main__":
    main()
