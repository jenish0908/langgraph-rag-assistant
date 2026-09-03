"""
The web app.

    streamlit run app.py

Why a UI matters here beyond looking nice: on CPU a question takes 20-90
seconds. A terminal that prints nothing for a minute feels broken. Streaming
the graph's progress turns dead time into visible work - and it doubles as the
best debugging view you have, because you watch the path the graph takes.
"""

import os
import time
import uuid

import streamlit as st

# Streamlit Cloud provides secrets via st.secrets, but llm.py reads plain
# environment variables so it works identically in a terminal, in Docker and
# on Streamlit Cloud. Bridge the two here - BEFORE importing anything that
# calls detect_provider(), or the provider is chosen before the key exists.
for _key in ("GROQ_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY",
             "LLM_PROVIDER", "CHECKPOINT_DB"):
    try:
        if _key in st.secrets and _key not in os.environ:
            os.environ[_key] = str(st.secrets[_key])
    except Exception:
        pass          # no secrets.toml locally - that is fine, we fall back

from llm import describe as describe_llm       # noqa: E402
from prompts import cite                        # noqa: E402
from step9_memory import build_graph, list_sources, thread, turn_input  # noqa: E402
from trace_log import TEXT_LOG, start_trace     # noqa: E402

st.set_page_config(page_title="Document Assistant", page_icon="📄")

# Human-readable labels for each node, shown live as it runs.
STEP_LABELS = {
    "contextualize": "Reading the conversation so far...",
    "classify": "Deciding whether to search the documents...",
    "chat_reply": "Replying...",
    "retrieve": "Searching the documents...",
    "grade_docs": "Checking whether the results actually answer this...",
    "rewrite_query": "No good match - rephrasing and retrying...",
    "generate": "Writing the answer...",
    "no_answer": "Nothing relevant found.",
}


@st.cache_resource(show_spinner="Loading documents and models (once)...")
def load_graph():
    """Built once per server, not once per message.

    @st.cache_resource is Streamlit's version of the lesson from Step 5:
    expensive setup happens once. Streamlit re-runs this entire script on
    every interaction, so without the cache the vector store would be rebuilt
    on every keystroke.
    """
    return build_graph(), list_sources()


graph, sources = load_graph()

# One conversation per browser session. The graph's checkpointer keys
# everything off this id, and it persists in memory.db.
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"web-{uuid.uuid4().hex[:8]}"
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar --------------------------------------------------------------
with st.sidebar:
    st.subheader("Documents")
    for s in sources:
        st.markdown(f"- `{s}`")

    st.subheader("How it works")
    st.markdown(
        "1. **contextualize** – resolve follow-ups (\"what about the A12?\")\n"
        "2. **classify** – does this need the docs at all?\n"
        "3. **retrieve** – find the 3 closest chunks\n"
        "4. **grade** – do they actually answer it?\n"
        "5. **rewrite** – if not, rephrase and retry (max 3)\n"
        "6. **generate** – answer using only those chunks"
    )

    st.subheader("Engine")
    st.caption(f"Language model: `{describe_llm()}`")
    st.caption("Search: `fastembed / bge-small-en-v1.5` (runs in-process)")

    st.divider()
    st.caption(f"Conversation: `{st.session_state.thread_id}`")
    st.caption(f"Full trace log: `logs/{TEXT_LOG.name}`")
    if st.button("New conversation"):
        st.session_state.thread_id = f"web-{uuid.uuid4().hex[:8]}"
        st.session_state.messages = []
        st.rerun()

# --- Chat history ---------------------------------------------------------
st.title("📄 Document Assistant")
st.caption("Ask about the documents in the sidebar. Follow-up questions work.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("trace"):
            with st.expander(f"How this was answered ({msg['elapsed']:.0f}s)"):
                st.code(msg["trace"], language="text")

# --- New message ----------------------------------------------------------
if question := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        status = st.empty()
        t0 = time.time()
        trace, answer, used_docs = [], "", []

        # Begin the full trace log for this question. Everything the nodes do
        # is written to logs/assistant.log - see trace_log.py.
        qtrace = start_trace(question, st.session_state.thread_id)

        # stream_mode="updates" yields after EACH node, so we can show live
        # progress and build a trace. Same debugging tool as Step 2, now
        # doing double duty as the UI.
        for chunk in graph.stream(
            turn_input(question),
            config=thread(st.session_state.thread_id),
            stream_mode="updates",
        ):
            for node, update in chunk.items():
                status.info(STEP_LABELS.get(node, node))
                elapsed = time.time() - t0

                if node == "contextualize":
                    q = update["standalone_question"]
                    if q.lower() != question.lower():
                        trace.append(f"[{elapsed:5.1f}s] contextualize -> {q!r}")
                    else:
                        trace.append(f"[{elapsed:5.1f}s] contextualize -> unchanged")
                elif node == "classify":
                    trace.append(f"[{elapsed:5.1f}s] classify -> {update['route']}")
                elif node == "retrieve":
                    ids = ", ".join(cite(d) for d in update["documents"])
                    trace.append(f"[{elapsed:5.1f}s] retrieve attempt "
                                 f"{update['attempts']}: {ids}")
                elif node == "grade_docs":
                    kept = update["relevant_docs"] or []
                    used_docs.extend(kept)
                    trace.append(f"[{elapsed:5.1f}s] grade -> kept {len(kept)} chunk(s)")
                elif node == "rewrite_query":
                    trace.append(f"[{elapsed:5.1f}s] rewrite -> {update['question']!r}")
                elif node in ("generate", "chat_reply", "no_answer"):
                    trace.append(f"[{elapsed:5.1f}s] {node}")
                    answer = update["answer"]

        elapsed = time.time() - t0
        qtrace.finish(answer)          # close the log entry, write the JSONL row
        status.empty()
        st.markdown(answer)

        # Show which chunks the answer was actually built from.
        if used_docs:
            seen = set()
            with st.expander(f"Sources ({len(used_docs)} chunk(s))"):
                for d in used_docs:
                    key = (d.metadata["source"], d.metadata["chunk_id"])
                    if key in seen:
                        continue
                    seen.add(key)
                    st.markdown(f"**{cite(d)}**")
                    st.text(d.page_content[:500].strip())

        trace_text = "\n".join(trace)
        with st.expander(f"How this was answered ({elapsed:.0f}s)"):
            st.code(trace_text, language="text")

    st.session_state.messages.append({
        "role": "assistant", "content": answer,
        "trace": trace_text, "elapsed": elapsed,
    })
