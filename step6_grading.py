"""
Step 6 - grade_docs and the first CONDITIONAL EDGE
==================================================
The graph stops being a straight line and starts making decisions.

    START -> [retrieve] -> [grade_docs] --relevant--> [generate] -> END
                                |
                           not relevant
                                v
                           [no_answer] -> END

Run it:   python step6_grading.py
"""

import time
from typing import TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from llm import get_llm
from prompts import (ANSWER_PROMPT, GRADE_PROMPT, format_docs,
                     number_docs, parse_verdicts)
from step4_embed_and_store import build_vector_store

K = 3

print("Building the vector store (once)...")
STORE = build_vector_store()
ANSWER_LLM = get_llm(max_tokens=250)
GRADE_LLM = get_llm(max_tokens=30)   # verdicts are ~12 tokens; cap hard
print("Ready.\n")


# ---------------------------------------------------------------------------
# THE STATE - two new fields since Step 5
# ---------------------------------------------------------------------------
class State(TypedDict):
    question: str
    documents: list[Document]        # everything retrieve found
    relevant_docs: list[Document]    # NEW: what survived grading
    answer: str


# ---------------------------------------------------------------------------
# THE NODES
# ---------------------------------------------------------------------------
def retrieve(state: State) -> dict:
    docs = STORE.similarity_search(state["question"], k=K)
    print("  [retrieve]   " + ", ".join(
        f"{d.metadata['source']}#{d.metadata['chunk_id']}" for d in docs))
    return {"documents": docs}


def grade_docs(state: State) -> dict:
    """Decide which retrieved chunks actually answer the question.

    ONE call for all K documents, not K calls. Prompt-reading dominates cost
    on this hardware, so K separate calls would mean K separate prompt reads.
    """
    docs = state["documents"]
    print("  [grade_docs] judging...", end="", flush=True)
    t0 = time.time()

    raw = GRADE_LLM.invoke(GRADE_PROMPT.format(
        context=number_docs(docs),
        question=state["question"],
    )).content

    verdicts = parse_verdicts(raw, len(docs))
    kept = [d for d, ok in zip(docs, verdicts) if ok]

    marks = " ".join(
        f"{d.metadata['source']}#{d.metadata['chunk_id']}:{'KEEP' if ok else 'drop'}"
        for d, ok in zip(docs, verdicts))
    print(f" {time.time() - t0:.1f}s -> {raw.strip()!r}")
    print(f"               {marks}")

    return {"relevant_docs": kept}


def generate(state: State) -> dict:
    """Answer using ONLY the chunks that survived grading."""
    print("  [generate]   answering...", end="", flush=True)
    t0 = time.time()
    response = ANSWER_LLM.invoke(ANSWER_PROMPT.format(
        context=format_docs(state["relevant_docs"]),
        question=state["question"],
    ))
    print(f" {time.time() - t0:.1f}s")
    return {"answer": response.content.strip()}


def no_answer(state: State) -> dict:
    """Nothing relevant was retrieved. Say so.

    A PLAIN PYTHON NODE - no LLM. It is instant, free, and cannot
    hallucinate. Worth noticing: the honest answer needs no intelligence.
    """
    print("  [no_answer]  nothing relevant survived grading")
    return {"answer": "I don't know based on the provided documents."}


# ---------------------------------------------------------------------------
# THE ROUTER - not a node!
#
# It takes state and returns the NAME of the next node. It must not modify
# state and should not do real work. grade_docs already did the thinking and
# wrote its verdict into state; the router only reads it.
#
# Keep the LLM call in the node and the `if` in the router. Mixing them makes
# both untestable.
# ---------------------------------------------------------------------------
def route_after_grading(state: State) -> str:
    if state["relevant_docs"]:
        return "generate"
    return "no_answer"


# ---------------------------------------------------------------------------
# THE GRAPH
# ---------------------------------------------------------------------------
builder = StateGraph(State)
builder.add_node("retrieve", retrieve)
builder.add_node("grade_docs", grade_docs)
builder.add_node("generate", generate)
builder.add_node("no_answer", no_answer)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "grade_docs")

# THE NEW PART: a conditional edge.
builder.add_conditional_edges(
    "grade_docs",              # after this node runs...
    route_after_grading,       # ...call this function to decide where to go
    ["generate", "no_answer"], # ...and it may only return one of these
)

builder.add_edge("generate", END)
builder.add_edge("no_answer", END)

graph = builder.compile()


# ---------------------------------------------------------------------------
# DEMO
# ---------------------------------------------------------------------------
def ask(question: str) -> dict:
    print(f"\n{'=' * 72}\nQ: {question}\n{'=' * 72}")
    t0 = time.time()
    result = graph.invoke({"question": question})
    print(f"\nA: {result['answer']}")
    print(f"\n   ({len(result['relevant_docs'])}/{len(result['documents'])} chunks kept, "
          f"{time.time() - t0:.1f}s total)")
    return result


if __name__ == "__main__":
    ask("How many paid leave days do I get per year?")
    ask("What is the warranty on the AtlasArm?")

    print("\n\n" + "#" * 72)
    print("# THE ONE THAT MATTERS: not in the documents.")
    print("# In Step 5 we HOPED the model would refuse. Now we CHECK.")
    print("#" * 72)
    ask("What is the wifi password in the office?")

    print("\n" + "=" * 72)
    print("Look at the [grade_docs] lines above. On the wifi question every")
    print("chunk was dropped, so the router sent the graph to no_answer and")
    print("the LLM was never asked to answer at all.")
    print()
    print("That is the difference between hoping and checking: the model")
    print("cannot hallucinate an answer it was never asked to produce.")
    print("=" * 72)
