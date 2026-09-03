"""
Step 5 - Wire retrieve + generate
=================================
Your first WORKING RAG. Two nodes in a straight line:

    START -> [retrieve] -> [generate] -> END

Everything runs locally: the embedding model from Step 4 and gemma3:4b
through Ollama. No API key, no network.

Run it:   python step5_first_rag.py
"""

import time
from typing import TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from llm import get_llm
from prompts import ANSWER_PROMPT as PROMPT
from prompts import format_docs
from step4_embed_and_store import build_vector_store

# How many chunks to retrieve. See Step 4: too few and you miss the answer,
# too many and it drowns among distractors (and you pay for every token).
K = 3


# ---------------------------------------------------------------------------
# BUILT ONCE, AT IMPORT - not inside a node.
#
# Put build_vector_store() inside retrieve() and you would re-read every file,
# re-chunk it and re-embed all 11 chunks on EVERY question. It would still
# work, which is exactly what makes that bug easy to ship - it just gets
# quietly slower as your document set grows.
#
# Principle: nodes do PER-REQUEST work. Expensive setup happens once.
# ---------------------------------------------------------------------------
print("Building the vector store (once)...")
STORE = build_vector_store()
LLM = get_llm(max_tokens=250)
print("Ready.\n")


# ---------------------------------------------------------------------------
# THE STATE
# ---------------------------------------------------------------------------
class State(TypedDict):
    question: str               # what the user asked
    documents: list[Document]   # what retrieve found -> generate reads this
    answer: str                 # what generate produced


# ---------------------------------------------------------------------------
# THE NODES
# ---------------------------------------------------------------------------
def retrieve(state: State) -> dict:
    """Find the K chunks closest in meaning to the question.

    Pure Python + vector math. No LLM, so it's instant and free.
    """
    docs = STORE.similarity_search(state["question"], k=K)
    print(f"  [retrieve] found {len(docs)} chunks: "
          + ", ".join(f"{d.metadata['source']}#{d.metadata['chunk_id']}" for d in docs))
    return {"documents": docs}


def generate(state: State) -> dict:
    """Answer the question using ONLY the retrieved chunks.

    Reads `documents` - a key it did not write. retrieve put it there.
    Nodes never call each other; they communicate through state.
    """
    print(f"  [generate] asking the model...", end="", flush=True)
    t0 = time.time()

    prompt = PROMPT.format(
        context=format_docs(state["documents"]),
        question=state["question"],
    )
    response = LLM.invoke(prompt)

    print(f" done in {time.time() - t0:.1f}s")
    return {"answer": response.content.strip()}


# ---------------------------------------------------------------------------
# THE GRAPH
# ---------------------------------------------------------------------------
builder = StateGraph(State)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

graph = builder.compile()


# ---------------------------------------------------------------------------
# DEMO
# ---------------------------------------------------------------------------
def ask(question: str) -> str:
    print(f"\n{'=' * 72}\nQ: {question}\n{'=' * 72}")
    t0 = time.time()
    result = graph.invoke({"question": question})
    print(f"\nA: {result['answer']}")
    print(f"\n   (total {time.time() - t0:.1f}s)")
    return result["answer"]


if __name__ == "__main__":
    # --- Questions the documents CAN answer -------------------------------
    ask("How many paid leave days do I get per year?")
    ask("How accurate is the AtlasArm A5?")

    # --- A question requiring info from BOTH documents --------------------
    ask("What is the hotel spending limit in London?")

    # --- The question the documents CANNOT answer -------------------------
    print("\n\n" + "#" * 72)
    print("# THE INTERESTING ONE - the answer is NOWHERE in our documents.")
    print("# Watch whether the model admits it, or invents something from")
    print("# the security chunk it was handed.")
    print("#" * 72)
    ask("What is the wifi password in the office?")

    print("\n" + "=" * 72)
    print("If it said 'I don't know based on the provided documents' - good,")
    print("but do not trust that. Retrieval handed it the security section")
    print("(passwords, MFA, hardware keys) and asked about a password. The")
    print("pull toward answering is strong, and it will NOT resist every time.")
    print()
    print("Step 6 replaces that hope with a check.")
    print("=" * 72)
