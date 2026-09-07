"""
The web app.

    streamlit run app.py

The chat is the main view. A "Show steps" button slides a panel in from the
right showing exactly what the graph did for the current question - every node
in execution order, what it produced, and how long it took.

Why that matters beyond looking nice: on CPU a question takes 20-90 seconds.
A UI that shows nothing for a minute feels broken. The panel also doubles as
the best debugging view there is - you can see whether a bad answer came from
retrieval or from grading, without opening a log.
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

from graph_view import (fact_for, ingestion_html, next_node,  # noqa: E402
                        output_html, query_html, styles)
from llm import describe as describe_llm                       # noqa: E402
from prompts import cite                                       # noqa: E402
from step9_memory import (build_graph, get_resources,           # noqa: E402
                          list_sources, thread, turn_input)
from trace_log import TEXT_LOG, start_trace                     # noqa: E402

st.set_page_config(page_title="Document Assistant", page_icon="📄",
                   layout="wide")

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


# ---------------------------------------------------------------------------
# STARTUP
# ---------------------------------------------------------------------------
# A cached MUTABLE container. Streamlit hands back the same dict on every
# rerun, so it survives like a cached value - but nothing is DRAWN inside a
# cached function, which is the point.
#
# WHY NOT draw inside @st.cache_resource? That was the first attempt and it
# raised CacheReplayClosureError on the SECOND run: on a cache hit Streamlit
# replays the elements the function drew last time, but the placeholder it
# drew into belonged to the previous run and no longer exists.
# Cache values, never drawing.
@st.cache_resource(show_spinner=False)
def ingest_record() -> dict:
    return {"facts": {}, "done": set(), "seconds": 0.0}


@st.cache_resource(show_spinner=False)
def compiled_graph():
    """Compile once - each call opens a SQLite connection for the checkpointer."""
    return build_graph()


record = ingest_record()

# Session state must exist before the drawer renders.
st.session_state.setdefault("thread_id", f"web-{uuid.uuid4().hex[:8]}")
st.session_state.setdefault("messages", [])
st.session_state.setdefault("drawer_open", False)
st.session_state.setdefault("last_runs", {})
st.session_state.setdefault("last_route", None)

st.markdown(styles(st.session_state.drawer_open), unsafe_allow_html=True)

# --- Header + the drawer toggle -------------------------------------------
head, toggle = st.columns([5, 1])
with head:
    st.title("📄 Document Assistant")
with toggle:
    st.write("")
    label = "✕  Hide steps" if st.session_state.drawer_open else "⚡  Show steps"
    if st.button(label, width="stretch"):
        st.session_state.drawer_open = not st.session_state.drawer_open
        st.rerun()

# --- The right-hand drawer -------------------------------------------------
# Rendered on EVERY run, open or shut. Closed just means translated off-screen
# by CSS - so the live updates below can write into it while it is hidden, and
# opening it mid-question shows the run already in progress.
drawer = st.container(key="steps_drawer")
with drawer:
    st.markdown("#### ⚡ Pipeline")
    st.caption("What the graph did for the most recent question.")
    live_flow = st.empty()
    live_flow.markdown(
        query_html(st.session_state.last_runs,
                   route=st.session_state.last_route,
                   finished=True),
        unsafe_allow_html=True)

    st.divider()
    st.markdown("###### Startup — ingestion, ran once")
    ingest_slot = st.empty()
    ingest_caption = st.empty()

# --- Ingestion (first script run in this process only) --------------------
if not record["done"]:
    current = {"stage": None}

    def on_stage(stage, fact):
        if fact is None:                      # stage STARTED
            current["stage"] = stage
        else:                                 # stage FINISHED
            record["facts"][stage] = fact
            record["done"].add(stage)
            current["stage"] = None
        ingest_slot.markdown(
            ingestion_html(current["stage"], record["done"], record["facts"]),
            unsafe_allow_html=True)

    t0 = time.time()
    with st.spinner("Starting up - reading and embedding your documents..."):
        get_resources(on_stage=on_stage)      # the expensive part
    record["seconds"] = time.time() - t0

ingest_slot.markdown(
    ingestion_html(None, record["done"], record["facts"]),
    unsafe_allow_html=True)
ingest_caption.caption(
    f"Ran once in {record['seconds']:.1f}s — not per question. "
    "Embedding is ~90% of it.")

graph = compiled_graph()
sources = list_sources()

# --- Sidebar --------------------------------------------------------------
with st.sidebar:
    st.subheader("Documents")
    for s in sources:
        st.markdown(f"- `{s}`")

    st.subheader("Engine")
    st.caption(f"Language model: `{describe_llm()}`")
    st.caption("Search: `fastembed / bge-small-en-v1.5`")

    st.divider()
    st.caption(f"Conversation: `{st.session_state.thread_id}`")
    st.caption(f"Trace log: `logs/{TEXT_LOG.name}`")
    if st.button("New conversation", width="stretch"):
        st.session_state.thread_id = f"web-{uuid.uuid4().hex[:8]}"
        st.session_state.messages = []
        st.session_state.last_runs = {}
        st.rerun()

st.caption("Ask about the documents in the sidebar. Follow-up questions work.")

# --- History ---------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("runs"):
            with st.expander(f"Steps for this answer ({msg['elapsed']:.0f}s)"):
                st.markdown(query_html(msg["runs"], route=msg.get("route"),
                                       finished=True),
                            unsafe_allow_html=True)

# --- New message ----------------------------------------------------------
if question := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        status = st.empty()

        runs, route, answer, used_docs = {}, None, "", []
        attempts = 0
        t0 = time.time()

        def paint(current=None, finished=False):
            """Redraw the drawer's graph. Safe whether it is open or shut."""
            live_flow.markdown(query_html(runs, current, route, finished),
                               unsafe_allow_html=True)

        paint("contextualize")
        status.info(STEP_LABELS["contextualize"])
        qtrace = start_trace(question, st.session_state.thread_id)

        # stream_mode="updates" yields after EACH node, so we can append a
        # step and repaint as the graph runs.
        for chunk in graph.stream(
            turn_input(question),
            config=thread(st.session_state.thread_id),
            stream_mode="updates",
        ):
            for node, update in chunk.items():
                elapsed = time.time() - t0
                detail = fact_for(node, update, question)

                if node == "classify":
                    route = update["route"]
                elif node == "retrieve":
                    attempts = update["attempts"]
                    detail += "  |  " + ", ".join(
                        cite(d) for d in update["documents"])
                elif node == "grade_docs":
                    kept = update["relevant_docs"] or []
                    used_docs.extend(kept)
                    if kept:
                        detail += ": " + ", ".join(cite(d) for d in kept)
                elif node in ("generate", "chat_reply", "no_answer"):
                    answer = update.get("answer", answer)

                # The matching trace_log event carries what never reaches
                # graph state: similarity scores, the grader's raw reply,
                # token counts. Pair it with the state update to build the
                # expandable "what did this step produce" panel.
                tev = next((e for e in reversed(qtrace.events)
                            if e.get("node") == node), {})
                runs.setdefault(node, []).append({
                    "detail": detail,
                    "seconds": tev.get("seconds", elapsed),
                    "output": output_html(node, update, tev, question),
                })

                upcoming = next_node(node, update, kept_total=len(used_docs),
                                     attempts=attempts)
                paint(upcoming)
                status.info(STEP_LABELS.get(upcoming, "Finishing up..."))

        elapsed = time.time() - t0
        qtrace.finish(answer)
        status.empty()
        paint(finished=True)                      # nothing running any more

        st.session_state.last_runs = runs
        st.session_state.last_route = route

        st.markdown(answer)

        if used_docs:
            seen = set()
            with st.expander(f"📎 Sources ({len(used_docs)} chunk(s))"):
                for d in used_docs:
                    key = (d.metadata["source"], d.metadata["chunk_id"])
                    if key in seen:
                        continue
                    seen.add(key)
                    st.markdown(f"**{cite(d)}**")
                    st.text(d.page_content[:500].strip())

    st.session_state.messages.append({
        "role": "assistant", "content": answer,
        "runs": runs, "route": route, "elapsed": elapsed,
    })
