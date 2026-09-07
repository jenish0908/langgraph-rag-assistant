"""
Step 9 - Conversation memory
============================
Two problems, both required. Solving only the first is the classic mistake.

  1. The graph forgets between calls        -> a CHECKPOINTER + thread_id
  2. Follow-ups are useless as search queries -> a CONTEXTUALIZE node

    START -> [contextualize] -> [classify] --chat--> [chat_reply] ----> END
                                     |
                                  search
                                     v
                                [retrieve] <----------------+
                                     |                      |
                                [grade_docs] --no--> [rewrite_query]
                                     |    \\
                                relevant    \\--exhausted--> [no_answer] -> END
                                     v
                                [generate] -> END

Run the CLI demo:   python step9_memory.py
Run the web app:    streamlit run app.py
"""

import os
import sqlite3
import time
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from llm import get_llm
from prompts import (ANSWER_PROMPT, GRADE_PROMPT, format_docs, number_docs,
                     parse_verdicts)
from step4_embed_and_store import build_vector_store
from trace_log import current as trace

K = 3
MAX_ATTEMPTS = 3
HISTORY_TURNS = 3          # how much conversation the contextualiser sees
# Where conversation history lives. Overridable because a hosting platform
# may give you only /tmp as writable - Streamlit Cloud does.
DB_PATH = Path(os.getenv("CHECKPOINT_DB", Path(__file__).parent / "memory.db"))


# ---------------------------------------------------------------------------
# THE RESET-ABLE REDUCER
#
# This is the trap that makes memory harder than it looks.
#
# `relevant_docs` uses a reducer so the retry loop can ACCUMULATE across laps
# (Step 7). But a checkpointer persists state across QUESTIONS too - so
# without a reset, question 2 would append to question 1's documents, and your
# context would slowly fill with chunks from unrelated questions.
#
# And you cannot reset an accumulating field by passing []: the reducer just
# appends an empty list. You need a sentinel the reducer understands.
# ---------------------------------------------------------------------------
def append_or_reset(old: list, new) -> list:
    """Append normally; treat None as 'clear this field'."""
    if new is None:
        return []
    return (old or []) + new


# ---------------------------------------------------------------------------
# LAZY SETUP
# ---------------------------------------------------------------------------
_RESOURCES = None


def get_resources(on_stage=None) -> dict:
    """Build the store and models once, on first use.

    on_stage is forwarded to build_vector_store so a UI can show ingestion
    progress. It only fires on the FIRST call - afterwards everything is
    already built, which is the whole point.
    """
    global _RESOURCES
    if _RESOURCES is None:
        _RESOURCES = {
            "store": build_vector_store(on_stage=on_stage),
            "answer_llm": get_llm(max_tokens=250),
            "grade_llm": get_llm(max_tokens=30),
            "rewrite_llm": get_llm(max_tokens=40),
            "classify_llm": get_llm(max_tokens=5),
            "chat_llm": get_llm(max_tokens=120),
            "context_llm": get_llm(max_tokens=40),
        }
    return _RESOURCES


def list_sources() -> list[str]:
    store = get_resources()["store"]
    return sorted({e["metadata"]["source"] for e in store.store.values()})


# ---------------------------------------------------------------------------
# THE STATE
# ---------------------------------------------------------------------------
class State(TypedDict):
    # PERSISTS across turns (that is the whole point of memory)
    messages: Annotated[list[BaseMessage], add_messages]

    # RESET every turn - see contextualize()
    user_message: str            # what the user literally typed
    standalone_question: str     # the same question, made self-contained
    question: str                # current search query (rewritten by the loop)
    route: str
    documents: list[Document]
    relevant_docs: Annotated[list[Document], append_or_reset]
    tried_queries: Annotated[list[str], append_or_reset]
    attempts: int
    answer: str


# ---------------------------------------------------------------------------
# PROMPTS
# ---------------------------------------------------------------------------
CONTEXTUALIZE_PROMPT = """Rewrite the user's latest message so it can be understood on its own,
without the conversation. Resolve pronouns and fill in anything implied.

If the message is already self-contained, repeat it unchanged.

Examples:
  History: "What is the payload of the A5?" / "5 kg."
  Latest:  "what about the A12?"
  Rewritten: "What is the payload of the AtlasArm A12?"

  History: "How much leave do I get?" / "21 days per year."
  Latest:  "how do I request it?"
  Rewritten: "How do I request paid leave?"

  History: (anything)
  Latest:  "What is the warranty on the arm?"
  Rewritten: "What is the warranty on the arm?"

Conversation so far:
{history}

Latest message: {question}

Reply with ONLY the rewritten question on one line. No explanation.

Rewritten:"""

CLASSIFY_PROMPT = """You decide whether a user message needs a search of the company's
internal documents (an employee handbook and a robot product FAQ).

Reply with exactly one word:
  SEARCH - the message asks for a FACT that would be written in those documents
  CHAT   - a greeting, thanks, small talk, or a question about YOU (the assistant)

Examples:
  "hello there"                  -> CHAT
  "what can you do?"             -> CHAT
  "what documents do you have?"  -> CHAT
  "thanks, that helps"           -> CHAT
  "how much leave do I get?"     -> SEARCH
  "is the arm IP67 rated?"       -> SEARCH

Message: {question}

Answer:"""

CHAT_PROMPT = """You are a document assistant. You answer questions about these files:
{sources}

Reply to the user in 2 sentences or fewer. Be brief and friendly.

Conversation so far:
{history}

User: {question}

Reply:"""

REWRITE_PROMPT = """Rewrite this question as a search query for a technical document database.

Make implicit ideas explicit: replace casual or situational wording with the
formal terms a manual or policy document would use.

Examples of the transformation:
  "if it falls over"      -> "collision damage"
  "money for my desk"     -> "home office equipment stipend"
  "till I'm permanent"    -> "probation period"

Question: {question}
Already tried (do not repeat these):
{tried}

Reply with ONLY the new search query on one line. No explanation.

New query:"""


def render_history(messages: list[BaseMessage], turns: int = HISTORY_TURNS) -> str:
    """Last few exchanges as plain text. Empty string when there is none."""
    recent = messages[-turns * 2:] if messages else []
    lines = []
    for m in recent:
        who = "User" if isinstance(m, HumanMessage) else "Assistant"
        lines.append(f"{who}: {m.content}")
    return "\n".join(lines)


def dedupe(docs: list[Document]) -> list[Document]:
    seen, out = set(), []
    for d in docs:
        key = (d.metadata["source"], d.metadata["chunk_id"])
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


# ---------------------------------------------------------------------------
# NODES
# ---------------------------------------------------------------------------
def contextualize(state: State) -> dict:
    """Start the turn: reset scratch state, then make the question standalone.

    Storing history does NOT fix retrieval. "what about the A12?" embedded as
    a search query is semantically almost nothing - no "payload", no "robot".
    It has to be rewritten into a self-contained question BEFORE retrieve
    ever sees it.
    """
    r = get_resources()
    user_msg = state["user_message"]
    history = render_history(state.get("messages", []))

    t0 = time.time()
    if not history:
        # First turn: nothing to resolve, so skip the LLM call entirely.
        standalone = user_msg
    else:
        standalone = r["context_llm"].invoke(CONTEXTUALIZE_PROMPT.format(
            history=history, question=user_msg)).content.strip().strip('"')

    trace().contextualize(
        seconds=time.time() - t0,
        history_len=len(state.get("messages", [])),
        standalone=standalone,
        rewritten=bool(history) and standalone.lower() != user_msg.lower(),
    )

    return {
        "messages": [HumanMessage(content=user_msg)],
        "standalone_question": standalone,
        "question": standalone,
        # --- reset per-turn scratch, or it leaks across the conversation ---
        "relevant_docs": None,      # sentinel -> append_or_reset clears it
        "tried_queries": None,      # same
        "documents": [],
        "attempts": 0,
        "answer": "",
    }


def classify(state: State) -> dict:
    r = get_resources()
    t0 = time.time()
    out = r["classify_llm"].invoke(
        CLASSIFY_PROMPT.format(question=state["standalone_question"])).content
    route = "search" if "SEARCH" in out.upper() else "chat"
    trace().classify(time.time() - t0, out, route)
    return {"route": route}


def chat_reply(state: State) -> dict:
    r = get_resources()
    t0 = time.time()
    reply = r["chat_llm"].invoke(CHAT_PROMPT.format(
        sources="\n".join(f"- {s}" for s in list_sources()),
        history=render_history(state["messages"][:-1]),   # exclude the message just added
        question=state["user_message"],
    )).content.strip()
    trace().chat_reply(time.time() - t0, reply)
    return {"answer": reply, "messages": [AIMessage(content=reply)]}


def retrieve(state: State) -> dict:
    r = get_resources()
    n = state.get("attempts", 0) + 1
    t0 = time.time()
    # _with_score so the log can show HOW close each chunk was, not just which.
    scored = r["store"].similarity_search_with_score(state["question"], k=K)
    docs = [d for d, _ in scored]
    trace().retrieve(time.time() - t0, n, state["question"], scored)
    return {"documents": docs, "attempts": n, "tried_queries": [state["question"]]}


def grade_docs(state: State) -> dict:
    r = get_resources()
    docs = state["documents"]
    t0 = time.time()
    raw = r["grade_llm"].invoke(GRADE_PROMPT.format(
        context=number_docs(docs), question=state["question"])).content
    verdicts = parse_verdicts(raw, len(docs))
    kept = [d for d, ok in zip(docs, verdicts) if ok]
    trace().grade(time.time() - t0, state["attempts"], raw, docs, verdicts)
    return {"relevant_docs": kept}


def rewrite_query(state: State) -> dict:
    r = get_resources()
    t0 = time.time()
    new_q = r["rewrite_llm"].invoke(REWRITE_PROMPT.format(
        question=state["standalone_question"],
        tried="\n".join(f"- {q}" for q in state["tried_queries"]),
    )).content.strip().strip('"')
    trace().rewrite(time.time() - t0, state["question"], new_q, state["tried_queries"])
    return {"question": new_q}


def generate(state: State) -> dict:
    r = get_resources()
    docs = dedupe(state["relevant_docs"])
    t0 = time.time()
    response = r["answer_llm"].invoke(ANSWER_PROMPT.format(
        context=format_docs(docs),
        question=state["standalone_question"],   # the resolved question
    ))
    answer = response.content.strip()
    trace().generate(time.time() - t0, docs, answer, response.response_metadata)
    return {"answer": answer, "messages": [AIMessage(content=answer)]}


def no_answer(state: State) -> dict:
    trace().no_answer(state["attempts"])
    msg = "I don't know based on the provided documents."
    return {"answer": msg, "messages": [AIMessage(content=msg)]}


# ---------------------------------------------------------------------------
# ROUTERS
# ---------------------------------------------------------------------------
def route_question(state: State) -> str:
    return "retrieve" if state["route"] == "search" else "chat_reply"


def route_after_grading(state: State) -> str:
    if state["relevant_docs"]:
        return "generate"
    if state["attempts"] >= MAX_ATTEMPTS:
        return "no_answer"
    return "rewrite_query"


# ---------------------------------------------------------------------------
# THE GRAPH
# ---------------------------------------------------------------------------
def make_checkpointer(db_path: Path = DB_PATH) -> SqliteSaver:
    """Conversation storage that survives a restart.

    check_same_thread=False because Streamlit serves requests from a thread
    pool. InMemorySaver would also work and needs no file - but then closing
    the app forgets everything.
    """
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        saver = SqliteSaver(conn)
        saver.setup()
        return saver
    except (sqlite3.Error, OSError) as e:
        # A read-only filesystem must not take the whole app down. Memory
        # then lasts only as long as the process, which is the correct
        # degradation: the app still works, it just forgets on restart.
        from langgraph.checkpoint.memory import InMemorySaver
        print(f"  note: cannot write {db_path} ({e}); "
              f"conversation memory will not survive a restart")
        return InMemorySaver()


def build_graph(checkpointer=None):
    builder = StateGraph(State)
    for name, fn in [("contextualize", contextualize), ("classify", classify),
                     ("chat_reply", chat_reply), ("retrieve", retrieve),
                     ("grade_docs", grade_docs), ("rewrite_query", rewrite_query),
                     ("generate", generate), ("no_answer", no_answer)]:
        builder.add_node(name, fn)

    builder.add_edge(START, "contextualize")
    builder.add_edge("contextualize", "classify")
    builder.add_conditional_edges("classify", route_question,
                                  ["retrieve", "chat_reply"])
    builder.add_edge("retrieve", "grade_docs")
    builder.add_conditional_edges("grade_docs", route_after_grading,
                                  ["generate", "no_answer", "rewrite_query"])
    builder.add_edge("rewrite_query", "retrieve")
    builder.add_edge("chat_reply", END)
    builder.add_edge("generate", END)
    builder.add_edge("no_answer", END)

    # The checkpointer is attached at COMPILE time, not build time.
    return builder.compile(checkpointer=checkpointer or make_checkpointer())


def turn_input(user_message: str) -> dict:
    """Input for one turn. Only the new message - the rest comes from the checkpoint."""
    return {"user_message": user_message}


def thread(thread_id: str) -> dict:
    """The config that selects WHICH conversation this belongs to."""
    return {"configurable": {"thread_id": thread_id}}


# ---------------------------------------------------------------------------
# CLI DEMO
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uuid

    from trace_log import JSONL_LOG, TEXT_LOG, start_trace

    graph = build_graph()
    tid = f"demo-{uuid.uuid4().hex[:8]}"   # fresh conversation each run
    cfg = thread(tid)

    conversation = [
        "What is the payload of the AtlasArm A5?",
        "what about the A12?",          # <- the follow-up that needs context
        "What is the wifi password in the office?",   # <- triggers the retry loop
        "thanks!",                      # <- small talk, should skip retrieval
    ]

    print(f"thread_id = {tid}")
    print(f"logging to {TEXT_LOG}\n")

    for msg in conversation:
        t = start_trace(msg, tid)              # <- begin the log for this question
        result = graph.invoke(turn_input(msg), config=cfg)
        t.finish(result["answer"])             # <- close it and write the JSONL row

    print(f"\nFull detail : {TEXT_LOG}")
    print(f"Structured  : {JSONL_LOG}")
