"""
Step 2 — Hello Graph
====================
Three fake nodes wired into a graph. No LLM, no documents, no API key.

The ONLY goal: see how state flows between nodes.

Run it:   python step2_hello_graph.py
"""

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph


# ---------------------------------------------------------------------------
# 1. THE STATE — the single dictionary that flows through the whole graph.
# ---------------------------------------------------------------------------
class State(TypedDict):
    # Default behaviour: each new value REPLACES the old one.
    text: str
    word_count: int
    summary: str

    # `Annotated[..., operator.add]` attaches a REDUCER.
    # For lists, `operator.add` means concatenate, so returning
    # {"log": ["hi"]} APPENDS instead of overwriting.
    log: Annotated[list[str], operator.add]


# ---------------------------------------------------------------------------
# 2. THE NODES — plain Python functions.
#
#    Contract:  take the whole state  ->  return ONLY the keys you changed.
# ---------------------------------------------------------------------------
def clean(state: State) -> dict:
    """Tidy up the incoming text. Demonstrates: OVERWRITING an existing key."""
    print("  [clean]  running...")
    cleaned = state["text"].strip().lower()
    return {
        "text": cleaned,              # replaces the old value
        "log": ["cleaned the text"],  # APPENDS, because of the reducer
    }


def count(state: State) -> dict:
    """Count the words. Demonstrates: WRITING A NEW key."""
    print("  [count]  running...")
    n = len(state["text"].split())
    return {
        "word_count": n,
        "log": [f"counted {n} words"],
    }


def report(state: State) -> dict:
    """Build a summary. Demonstrates: READING keys that earlier nodes wrote."""
    print("  [report] running...")
    line = f"The text has {state['word_count']} words: {state['text']!r}"
    return {
        "summary": line,
        "log": ["built the report"],
    }


# ---------------------------------------------------------------------------
# 3. BUILD THE GRAPH — declare the boxes and the arrows.
# ---------------------------------------------------------------------------
builder = StateGraph(State)

# Register each function as a node under a string name.
builder.add_node("clean", clean)
builder.add_node("count", count)
builder.add_node("report", report)

# Connect them:  START -> clean -> count -> report -> END
builder.add_edge(START, "clean")
builder.add_edge("clean", "count")
builder.add_edge("count", "report")
builder.add_edge("report", END)

# COMPILE: validate everything and freeze it into a runnable object.
graph = builder.compile()


# ---------------------------------------------------------------------------
# 4. RUN IT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    starting_state = {
        "text": "   LangGraph Makes Agents EASY   ",
        "log": [],  # reducers append to this, so it must start as a list
    }

    # --- Way 1: invoke() -> run everything, give me the FINAL state -------
    print("=" * 62)
    print("RUN 1 - invoke():  runs to completion, returns final state")
    print("=" * 62)

    final = graph.invoke(starting_state)

    print("\nFinal state:")
    for key, value in final.items():
        print(f"  {key:12} = {value!r}")

    # --- Way 2: stream() -> show me what EACH node returned ---------------
    print("\n" + "=" * 62)
    print("RUN 2 - stream():  peek at each node's output as it happens")
    print("=" * 62)
    print("(This is your debugger. It shows WHO changed WHAT.)\n")

    for chunk in graph.stream(starting_state, stream_mode="updates"):
        for node_name, update in chunk.items():
            print(f"  node {node_name!r} returned -> {update}")

    # --- The lesson -------------------------------------------------------
    print("\n" + "=" * 62)
    print("NOTICE:")
    print("  - Each node returned only 2 keys, never the whole state.")
    print("  - 'text' was REPLACED by clean  (default reducer).")
    print("  - 'log' ACCUMULATED all 3 entries (operator.add reducer).")
    print("  - 'report' could read 'word_count', which 'count' wrote.")
    print("=" * 62)
