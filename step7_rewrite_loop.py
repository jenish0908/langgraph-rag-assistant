"""
Step 7 - rewrite_query and the CYCLE
====================================
The graph gains an edge that points BACKWARDS.

                    +-------------------------------+
                    v                               |
    START -> [retrieve] -> [grade_docs] -> [rewrite_query]
                                 |
                             relevant
                                 v
                            [generate] -> END
                                 |
                        attempts exhausted
                                 v
                            [no_answer] -> END

Run it:   python step7_rewrite_loop.py
"""

import operator
import time
from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from llm import get_llm
from prompts import (ANSWER_PROMPT, GRADE_PROMPT, format_docs,
                     number_docs, parse_verdicts)
from step4_embed_and_store import build_vector_store

K = 3
MAX_ATTEMPTS = 3          # the loop guard. Without this it runs forever.

print("Building the vector store (once)...")
STORE = build_vector_store()
ANSWER_LLM = get_llm(max_tokens=250)
GRADE_LLM = get_llm(max_tokens=30)
REWRITE_LLM = get_llm(max_tokens=40)
print("Ready.\n")


# ---------------------------------------------------------------------------
# THE STATE - now built for a LOOP, not a straight line
# ---------------------------------------------------------------------------
class State(TypedDict):
    original_question: str   # what the user actually asked - never changes
    question: str            # the CURRENT search query - rewritten each lap
    documents: list[Document]        # what the latest retrieve found (replaced)

    # ACCUMULATES across laps. This is the Step 2 reducer finally earning its
    # keep: retrieve runs up to 3 times, and without operator.add each lap
    # would DESTROY what the previous laps found.
    relevant_docs: Annotated[list[Document], operator.add]
    tried_queries: Annotated[list[str], operator.add]

    attempts: int            # the loop counter
    answer: str


# ---------------------------------------------------------------------------
# THE REWRITE PROMPT
#
# The key instruction is "make implicit ideas explicit". A user asks in
# situational language ("if it falls over"); documents are written in formal
# language ("collision damage"). The examples matter far more than the
# description - a 4B model imitates patterns much better than it follows
# abstract instructions.
# ---------------------------------------------------------------------------
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
    """Drop repeats accumulated across laps.

    Lap 2 often retrieves some of the same chunks as lap 1. Because
    relevant_docs uses operator.add, those duplicates pile up - and duplicated
    context wastes the prompt budget that dominates our runtime.
    """
    seen, out = set(), []
    for d in docs:
        key = (d.metadata["source"], d.metadata["chunk_id"])
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


# ---------------------------------------------------------------------------
# THE NODES
# ---------------------------------------------------------------------------
def retrieve(state: State) -> dict:
    """Search with the CURRENT query. Runs 1-3 times per question."""
    n = state.get("attempts", 0) + 1
    docs = STORE.similarity_search(state["question"], k=K)
    print(f"  [retrieve]     attempt {n}: {state['question']!r}")
    print("                 -> " + ", ".join(
        f"{d.metadata['source'].split('.')[0][:4]}#{d.metadata['chunk_id']}" for d in docs))
    return {
        "documents": docs,
        "attempts": n,
        "tried_queries": [state["question"]],
    }


def grade_docs(state: State) -> dict:
    """Judge the CURRENT results against the CURRENT query."""
    docs = state["documents"]
    print("  [grade_docs]   judging...", end="", flush=True)
    t0 = time.time()
    raw = GRADE_LLM.invoke(GRADE_PROMPT.format(
        context=number_docs(docs), question=state["question"])).content
    verdicts = parse_verdicts(raw, len(docs))
    kept = [d for d, ok in zip(docs, verdicts) if ok]
    print(f" {time.time() - t0:.1f}s -> {raw.strip()!r}  ({len(kept)}/{len(docs)} kept)")
    return {"relevant_docs": kept}      # APPENDS, thanks to the reducer


def rewrite_query(state: State) -> dict:
    """Rephrase the question and send the graph back to retrieve."""
    print("  [rewrite]      rephrasing...", end="", flush=True)
    t0 = time.time()
    tried = "\n".join(f"- {q}" for q in state["tried_queries"])
    new_q = REWRITE_LLM.invoke(REWRITE_PROMPT.format(
        question=state["original_question"], tried=tried)).content.strip().strip('"')
    print(f" {time.time() - t0:.1f}s -> {new_q!r}")
    return {"question": new_q}


def generate(state: State) -> dict:
    """Answer the ORIGINAL question using every relevant chunk found in ANY lap."""
    docs = dedupe(state["relevant_docs"])
    print(f"  [generate]     answering from {len(docs)} chunk(s)...", end="", flush=True)
    t0 = time.time()
    response = ANSWER_LLM.invoke(ANSWER_PROMPT.format(
        context=format_docs(docs),
        question=state["original_question"],   # the ORIGINAL, not the rewrite
    ))
    print(f" {time.time() - t0:.1f}s")
    return {"answer": response.content.strip()}


def no_answer(state: State) -> dict:
    print(f"  [no_answer]    gave up after {state['attempts']} attempts")
    return {"answer": "I don't know based on the provided documents."}


# ---------------------------------------------------------------------------
# THE ROUTER - now with THREE destinations, one of which loops backwards
#
# A cycle with no exit is an infinite loop.
# A cycle with an exit is a retry policy.
# The counter is the entire difference.
# ---------------------------------------------------------------------------
def route_after_grading(state: State) -> str:
    if state["relevant_docs"]:
        return "generate"                 # got what we need - leave the loop
    if state["attempts"] >= MAX_ATTEMPTS:
        return "no_answer"                # tried enough - leave the loop
    return "rewrite_query"                # go around again


# ---------------------------------------------------------------------------
# THE GRAPH
# ---------------------------------------------------------------------------
builder = StateGraph(State)
builder.add_node("retrieve", retrieve)
builder.add_node("grade_docs", grade_docs)
builder.add_node("rewrite_query", rewrite_query)
builder.add_node("generate", generate)
builder.add_node("no_answer", no_answer)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "grade_docs")
builder.add_conditional_edges(
    "grade_docs",
    route_after_grading,
    ["generate", "no_answer", "rewrite_query"],
)

# THE CYCLE: this edge points BACKWARDS to a node that already ran.
builder.add_edge("rewrite_query", "retrieve")

builder.add_edge("generate", END)
builder.add_edge("no_answer", END)

graph = builder.compile()


# ---------------------------------------------------------------------------
# DEMO
# ---------------------------------------------------------------------------
def ask(question: str) -> dict:
    print(f"\n{'=' * 74}\nQ: {question}\n{'=' * 74}")
    t0 = time.time()
    result = graph.invoke({
        "original_question": question,
        "question": question,
        "attempts": 0,
    })
    print(f"\nA: {result['answer']}")
    print(f"\n   attempts: {result['attempts']}   "
          f"queries tried: {result['tried_queries']}")
    print(f"   {time.time() - t0:.1f}s total")
    return result


if __name__ == "__main__":
    print("#" * 74)
    print("# 1. HAPPY PATH - succeeds on attempt 1, the loop never fires")
    print("#" * 74)
    ask("How many paid leave days do I get per year?")

    print("\n\n" + "#" * 74)
    print("# 2. THE LOOP FIRES - attempt 1 fails, a rewrite is tried")
    print("#" * 74)
    ask("Can the robot work in a paint booth?")

    print("\n\n" + "#" * 74)
    print("# 3. THE GUARD HOLDS - genuinely unanswerable, so it must give up")
    print("#    Watch attempts reach 3 and stop. Without MAX_ATTEMPTS this")
    print("#    would loop forever.")
    print("#" * 74)
    ask("What is the wifi password in the office?")
