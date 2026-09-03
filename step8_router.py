"""
Step 8 - the route node, and a module that is SAFE TO IMPORT
============================================================
The finished graph:

    START -> [classify] --- chat ---> [chat_reply] ------------> END
                 |
              search
                 v
            [retrieve] <---------------------+
                 |                           |
            [grade_docs] --no--> [rewrite_query]   (up to MAX_ATTEMPTS)
                 |    \\
              relevant  \\-- attempts exhausted --> [no_answer] -> END
                 v
            [generate] -> END

Run the CLI demo:   python step8_router.py
Run the web app:    streamlit run app.py
"""

import operator
import time
from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from llm import get_llm
from prompts import (ANSWER_PROMPT, GRADE_PROMPT, format_docs, number_docs,
                     parse_verdicts)
from step4_embed_and_store import build_vector_store

K = 3
MAX_ATTEMPTS = 3


# ---------------------------------------------------------------------------
# LAZY SETUP - this module is safe to import.
#
# Steps 5, 6 and 7 all built the vector store at module level. That made them
# expensive (and, twice, accidentally DOUBLE-expensive) to import. We hit that
# bug twice; the third time we fix the shape instead of the symptom.
#
# Nothing here runs until someone actually calls get_resources().
# ---------------------------------------------------------------------------
_RESOURCES = None


def get_resources() -> dict:
    """Build the store and models once, on first use."""
    global _RESOURCES
    if _RESOURCES is None:
        store = build_vector_store()
        _RESOURCES = {
            "store": store,
            "answer_llm": get_llm(max_tokens=250),
            "grade_llm": get_llm(max_tokens=30),
            "rewrite_llm": get_llm(max_tokens=40),
            "classify_llm": get_llm(max_tokens=5),
            "chat_llm": get_llm(max_tokens=120),
        }
    return _RESOURCES


def list_sources() -> list[str]:
    """Which files the assistant can see - used by the chat prompt and the UI."""
    store = get_resources()["store"]
    return sorted({entry["metadata"]["source"] for entry in store.store.values()})


# ---------------------------------------------------------------------------
# THE STATE
# ---------------------------------------------------------------------------
class State(TypedDict):
    original_question: str
    question: str
    route: str                       # NEW: "search" or "chat"
    documents: list[Document]
    relevant_docs: Annotated[list[Document], operator.add]
    tried_queries: Annotated[list[str], operator.add]
    attempts: int
    answer: str


# ---------------------------------------------------------------------------
# PROMPTS
# ---------------------------------------------------------------------------
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

# Few-shot examples were worth +3s of prompt: accuracy went from 8/9 to 10/10.
# The classifier costs ~7s but saves ~80s on every non-question, so it pays for
# itself the first time someone says "thanks".

CHAT_PROMPT = """You are a document assistant. You answer questions about these files:
{sources}

Reply to the user in 2 sentences or fewer. Be brief and friendly. If they seem
to want information, invite them to ask a specific question about those files.

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
def classify(state: State) -> dict:
    """Does this message need the documents at all?

    Without this node, "hi" costs ~80 seconds: retrieve, grade, rewrite,
    retrieve, grade, rewrite, retrieve, grade, give up.
    """
    r = get_resources()
    out = r["classify_llm"].invoke(
        CLASSIFY_PROMPT.format(question=state["original_question"])).content.upper()
    route = "search" if "SEARCH" in out else "chat"
    print(f"  [classify]     -> {route}")
    return {"route": route}


def chat_reply(state: State) -> dict:
    """Answer small talk without touching the documents."""
    r = get_resources()
    print("  [chat_reply]   replying...", end="", flush=True)
    t0 = time.time()
    reply = r["chat_llm"].invoke(CHAT_PROMPT.format(
        sources="\n".join(f"- {s}" for s in list_sources()),
        question=state["original_question"],
    )).content.strip()
    print(f" {time.time() - t0:.1f}s")
    return {"answer": reply}


def retrieve(state: State) -> dict:
    r = get_resources()
    n = state.get("attempts", 0) + 1
    docs = r["store"].similarity_search(state["question"], k=K)
    print(f"  [retrieve]     attempt {n}: {state['question']!r}")
    return {"documents": docs, "attempts": n, "tried_queries": [state["question"]]}


def grade_docs(state: State) -> dict:
    r = get_resources()
    docs = state["documents"]
    print("  [grade_docs]   judging...", end="", flush=True)
    t0 = time.time()
    raw = r["grade_llm"].invoke(GRADE_PROMPT.format(
        context=number_docs(docs), question=state["question"])).content
    verdicts = parse_verdicts(raw, len(docs))
    kept = [d for d, ok in zip(docs, verdicts) if ok]
    print(f" {time.time() - t0:.1f}s -> {raw.strip()!r} ({len(kept)}/{len(docs)} kept)")
    return {"relevant_docs": kept}


def rewrite_query(state: State) -> dict:
    r = get_resources()
    print("  [rewrite]      rephrasing...", end="", flush=True)
    t0 = time.time()
    new_q = r["rewrite_llm"].invoke(REWRITE_PROMPT.format(
        question=state["original_question"],
        tried="\n".join(f"- {q}" for q in state["tried_queries"]),
    )).content.strip().strip('"')
    print(f" {time.time() - t0:.1f}s -> {new_q!r}")
    return {"question": new_q}


def generate(state: State) -> dict:
    r = get_resources()
    docs = dedupe(state["relevant_docs"])
    print(f"  [generate]     from {len(docs)} chunk(s)...", end="", flush=True)
    t0 = time.time()
    response = r["answer_llm"].invoke(ANSWER_PROMPT.format(
        context=format_docs(docs),
        question=state["original_question"],
    ))
    print(f" {time.time() - t0:.1f}s")
    return {"answer": response.content.strip()}


def no_answer(state: State) -> dict:
    print(f"  [no_answer]    gave up after {state['attempts']} attempts")
    return {"answer": "I don't know based on the provided documents."}


# ---------------------------------------------------------------------------
# ROUTERS - pure functions. No work, no LLM, no state changes.
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
def build_graph():
    builder = StateGraph(State)
    for name, fn in [("classify", classify), ("chat_reply", chat_reply),
                     ("retrieve", retrieve), ("grade_docs", grade_docs),
                     ("rewrite_query", rewrite_query), ("generate", generate),
                     ("no_answer", no_answer)]:
        builder.add_node(name, fn)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", route_question,
                                  ["retrieve", "chat_reply"])
    builder.add_edge("retrieve", "grade_docs")
    builder.add_conditional_edges("grade_docs", route_after_grading,
                                  ["generate", "no_answer", "rewrite_query"])
    builder.add_edge("rewrite_query", "retrieve")   # the cycle
    builder.add_edge("chat_reply", END)
    builder.add_edge("generate", END)
    builder.add_edge("no_answer", END)
    return builder.compile()


def initial_state(question: str) -> dict:
    """The starting state for one question. Used by the CLI and the web app."""
    return {"original_question": question, "question": question, "attempts": 0}


# ---------------------------------------------------------------------------
# CLI DEMO
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    graph = build_graph()

    for q in ["hi there",
              "what can you do?",
              "How many paid leave days do I get per year?",
              "What is the wifi password in the office?"]:
        print(f"\n{'=' * 74}\nQ: {q}\n{'=' * 74}")
        t0 = time.time()
        result = graph.invoke(initial_state(q))
        print(f"\nA: {result['answer']}")
        print(f"   ({result['route']} path, {time.time() - t0:.1f}s)")
