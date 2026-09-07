"""
The web app - with live graph visualisation.

    streamlit run app.py

Two diagrams show where the system is at any moment:

  1. INGESTION, once at startup:  load -> chunk -> embed -> store
  2. QUERY, per question:         the 8-node graph, lighting up as it runs

Why this matters beyond looking nice: on CPU a question takes 20-90 seconds.
A UI that shows nothing for a minute feels broken. Watching the actual node
light up turns dead time into visible work - and it is the best debugging
view you have, because you see the path the graph took and what each step
produced.
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

from graph_view import (fact_for, ingestion_dot, next_node,  # noqa: E402
                        query_dot)
from llm import describe as describe_llm                      # noqa: E402
from prompts import cite                                      # noqa: E402
from step9_memory import (build_graph, get_resources,          # noqa: E402
                          list_sources, thread, turn_input)
from trace_log import TEXT_LOG, start_trace                    # noqa: E402

st.set_page_config(page_title="Document Assistant", page_icon="📄",
                   layout="wide")

# What to show in the status line while each node runs.
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
# STARTUP: build everything once, drawing the ingestion pipeline as it goes
# ---------------------------------------------------------------------------
# A cached MUTABLE container. Streamlit hands back the same dict on every
# rerun, so it survives like a cached value - but nothing is drawn inside a
# cached function, which is the point.
#
# WHY NOT just draw inside @st.cache_resource? That was the first attempt and
# it raised CacheReplayClosureError on the SECOND run:
#
#     While running bootstrap(), a streamlit element is called on some layout
#     block created outside the function. This is incompatible with replaying
#     the cached effect of that element.
#
# On a cache HIT Streamlit replays the elements the function drew last time -
# but our placeholder belonged to the previous run and no longer exists.
# Caching a value is fine; caching drawing is not.
@st.cache_resource(show_spinner=False)
def ingest_record() -> dict:
    return {"facts": {}, "done": set(), "seconds": 0.0}


@st.cache_resource(show_spinner=False)
def compiled_graph():
    """Compile once - each call opens a SQLite connection for the checkpointer."""
    return build_graph()


st.title("📄 Document Assistant")

record = ingest_record()

# The FIRST script run in this process does the real ingestion and draws it
# live. Later runs find `done` already populated and skip straight through -
# get_resources() is itself a singleton, so it returns instantly.
if not record["done"]:
    live = st.empty()
    current = {"stage": None}

    def on_stage(stage, fact):
        if fact is None:                      # stage STARTED
            current["stage"] = stage
        else:                                 # stage FINISHED
            record["facts"][stage] = fact
            record["done"].add(stage)
            current["stage"] = None
        live.graphviz_chart(
            ingestion_dot(current["stage"], record["done"], record["facts"]),
            width='stretch')

    t0 = time.time()
    with st.spinner("Starting up - reading and embedding your documents..."):
        get_resources(on_stage=on_stage)      # the expensive part
    record["seconds"] = time.time() - t0
    live.empty()                              # it reappears in the expander below

graph = compiled_graph()
sources = list_sources()
boot = record

with st.expander(f"⚙️ Ingestion pipeline — ran once at startup "
                 f"({boot['seconds']:.1f}s)"):
    st.graphviz_chart(ingestion_dot(None, boot["done"], boot["facts"]),
                      width='stretch')
    st.caption(
        "This happens once per server, not per question. Embedding is ~90% "
        "of it. Documents are read from `docs/`, split into overlapping "
        "chunks, turned into 384-dimension vectors, and kept in a searchable "
        "store."
    )

# One conversation per browser session; the checkpointer keys off this id.
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"web-{uuid.uuid4().hex[:8]}"
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar --------------------------------------------------------------
with st.sidebar:
    st.subheader("Documents")
    for s in sources:
        st.markdown(f"- `{s}`")

    st.subheader("Engine")
    st.caption(f"Language model: `{describe_llm()}`")
    st.caption("Search: `fastembed / bge-small-en-v1.5`")

    st.subheader("Legend")
    st.markdown(
        "🟡 running now  \n"
        "🟢 finished  \n"
        "⚪ not reached  \n"
        "🔴 gave up  \n"
        "*dashed edge* = the retry cycle"
    )

    st.divider()
    st.caption(f"Conversation: `{st.session_state.thread_id}`")
    st.caption(f"Full trace log: `logs/{TEXT_LOG.name}`")
    if st.button("New conversation"):
        st.session_state.thread_id = f"web-{uuid.uuid4().hex[:8]}"
        st.session_state.messages = []
        st.rerun()

st.caption("Ask about the documents in the sidebar. Follow-up questions work.")

# --- Replay history -------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("dot"):
            with st.expander(f"🔎 Path through the graph ({msg['elapsed']:.0f}s)"):
                left, right = st.columns([3, 2])
                with left:
                    st.graphviz_chart(msg["dot"], width='stretch')
                with right:
                    st.code(msg["steps"], language="text")

# --- New message ----------------------------------------------------------
if question := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        status = st.empty()
        diagram_col, steps_col = st.columns([3, 2])
        diagram = diagram_col.empty()
        step_log = steps_col.empty()

        # Live state for the diagram
        done, facts, failed = set(), {}, set()
        route, steps, answer, used_docs = None, [], "", []
        attempts = 0

        diagram.graphviz_chart(query_dot(current="contextualize"),
                               width='stretch')
        status.info(STEP_LABELS["contextualize"])

        t0 = time.time()
        qtrace = start_trace(question, st.session_state.thread_id)

        # stream_mode="updates" yields after EACH node, so we can redraw the
        # diagram and append a step line as the graph runs.
        for chunk in graph.stream(
            turn_input(question),
            config=thread(st.session_state.thread_id),
            stream_mode="updates",
        ):
            for node, update in chunk.items():
                elapsed = time.time() - t0
                done.add(node)
                facts[node] = fact_for(node, update, question)
                if node == "retrieve":
                    attempts = update["attempts"]

                # --- detail line for the step log ------------------------
                if node == "contextualize":
                    q = update["standalone_question"]
                    detail = (f"-> {q!r}" if q.lower() != question.lower()
                              else "already self-contained")
                elif node == "classify":
                    route = update["route"]
                    detail = f"-> {route}"
                elif node == "retrieve":
                    ids = ", ".join(cite(d) for d in update["documents"])
                    detail = f"attempt {update['attempts']}: {ids}"
                elif node == "grade_docs":
                    kept = update["relevant_docs"] or []
                    used_docs.extend(kept)
                    detail = (f"kept {len(kept)}: "
                              + (", ".join(cite(d) for d in kept) or "nothing"))
                elif node == "rewrite_query":
                    detail = f"-> {update['question']!r}"
                else:
                    detail = ""
                    answer = update.get("answer", answer)
                    if node == "no_answer":
                        failed.add(node)

                steps.append(f"[{elapsed:5.1f}s] {node}\n           {detail}"
                             if detail else f"[{elapsed:5.1f}s] {node}")
                step_log.code("\n".join(steps), language="text")

                # --- redraw with the NEXT node highlighted ---------------
                upcoming = next_node(node, update,
                                     kept_total=len(used_docs),
                                     attempts=attempts)
                diagram.graphviz_chart(
                    query_dot(upcoming, done, facts, route, failed),
                    width='stretch')
                status.info(STEP_LABELS.get(upcoming, "Working..."))

        elapsed = time.time() - t0
        qtrace.finish(answer)
        status.empty()

        final_dot = query_dot(None, done, facts, route, failed)
        diagram.graphviz_chart(final_dot, width='stretch')
        steps_text = "\n".join(steps)
        step_log.code(steps_text, language="text")

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
        "dot": final_dot, "steps": steps_text, "elapsed": elapsed,
    })
