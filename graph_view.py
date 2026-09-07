"""
Live graph visualisation.

Builds Graphviz DOT strings that show WHERE THE SYSTEM IS RIGHT NOW - which
node is executing, which have finished, which path the router took, and what
each one produced.

Two pipelines get drawn:

  1. INGESTION (runs once at startup)
         load -> chunk -> embed -> store

  2. QUERY (runs per question)
         contextualize -> classify -> {chat_reply | retrieve -> grade -> ...}

Streamlit renders DOT client-side with dagre-d3, so nothing needs the
Graphviz system binary - only the `graphviz` pip package.
"""

# --- Palette ---------------------------------------------------------------
# One colour per state, used by both pipelines so the two diagrams read the
# same way.
PENDING = ("#f8f9fa", "#ced4da", "#868e96")   # fill, border, text
RUNNING = ("#fff3bf", "#f59f00", "#663c00")   # amber - happening NOW
DONE = ("#d3f9d8", "#37b24d", "#1b4332")      # green - finished
SKIPPED = ("#f1f3f5", "#dee2e6", "#adb5bd")   # grey - path not taken
FAILED = ("#ffe3e3", "#f03e3e", "#c92a2a")    # red - gave up / nothing found

STATE_COLOURS = {
    "pending": PENDING,
    "running": RUNNING,
    "done": DONE,
    "skipped": SKIPPED,
    "failed": FAILED,
}


def _node(name: str, label: str, state: str = "pending") -> str:
    """One DOT node, coloured by state.

    dagre-d3 supports a useful subset of DOT. HTML-ish labels are not
    reliable, so multi-line labels use \\n escapes instead.
    """
    fill, border, text = STATE_COLOURS.get(state, PENDING)
    pen = 2.5 if state == "running" else 1.2
    safe = label.replace('"', "'")
    return (f'  {name} [label="{safe}", style="filled,rounded", shape=box, '
            f'fillcolor="{fill}", color="{border}", fontcolor="{text}", '
            f'penwidth={pen}, fontname="Helvetica", fontsize=11];')


def _edge(a: str, b: str, label: str = "", active: bool = False,
          back: bool = False) -> str:
    colour = "#37b24d" if active else "#ced4da"
    pen = 2.2 if active else 1.0
    style = "dashed" if back else "solid"
    lbl = f', label="{label}", fontsize=9, fontcolor="#868e96"' if label else ""
    return (f'  {a} -> {b} [color="{colour}", penwidth={pen}, '
            f'style={style}{lbl}];')


# ---------------------------------------------------------------------------
# 1. INGESTION PIPELINE
# ---------------------------------------------------------------------------
INGEST_STAGES = ["load", "chunk", "embed", "store"]

INGEST_LABELS = {
    "load": "1. LOAD FILES\\nread docs/ into Documents",
    "chunk": "2. CHUNK\\nsplit into overlapping pieces",
    "embed": "3. EMBED\\ntext -> 384-dim vectors",
    "store": "4. STORE\\nsearchable vector store",
}


def ingestion_dot(current: str | None = None, done: set[str] | None = None,
                  facts: dict[str, str] | None = None) -> str:
    """Draw the startup pipeline.

    current: the stage running right now (or None when finished)
    done:    stages already completed
    facts:   stage -> a short result line, e.g. {"chunk": "13 chunks"}
    """
    done = done or set()
    facts = facts or {}

    lines = ['digraph ingest {', '  rankdir=LR;', '  bgcolor="transparent";',
             '  node [margin="0.18,0.10"];']
    for stage in INGEST_STAGES:
        label = INGEST_LABELS[stage]
        if stage in facts:
            label += f"\\n{facts[stage]}"
        state = ("running" if stage == current
                 else "done" if stage in done else "pending")
        lines.append(_node(stage, label, state))

    for a, b in zip(INGEST_STAGES, INGEST_STAGES[1:]):
        lines.append(_edge(a, b, active=(a in done)))
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. QUERY GRAPH
# ---------------------------------------------------------------------------
QUERY_LABELS = {
    "contextualize": "contextualize\\nresolve follow-ups",
    "classify": "classify\\nneed the docs?",
    "chat_reply": "chat_reply\\nsmall talk",
    "retrieve": "retrieve\\nfind k nearest chunks",
    "grade_docs": "grade_docs\\ndo they answer it?",
    "rewrite_query": "rewrite_query\\nrephrase & retry",
    "generate": "generate\\nanswer + citations",
    "no_answer": "no_answer\\nadmit we don't know",
}


def query_dot(current: str | None = None,
              done: set[str] | None = None,
              facts: dict[str, str] | None = None,
              route: str | None = None,
              failed: set[str] | None = None) -> str:
    """Draw the per-question graph.

    current: node executing right now
    done:    nodes that have finished
    facts:   node -> short result, e.g. {"grade_docs": "kept 2/3"}
    route:   "search" or "chat" once classify has decided - greys out the
             branch that was not taken
    failed:  nodes to paint red (no_answer)
    """
    done = done or set()
    facts = facts or {}
    failed = failed or set()

    def state_of(name: str) -> str:
        if name in failed:
            return "failed"
        if name == current:
            return "running"
        if name in done:
            return "done"
        # Grey out the branch the router did not take.
        if route == "search" and name == "chat_reply":
            return "skipped"
        if route == "chat" and name in {"retrieve", "grade_docs",
                                        "rewrite_query", "generate",
                                        "no_answer"}:
            return "skipped"
        return "pending"

    lines = ['digraph query {', '  rankdir=TB;', '  bgcolor="transparent";',
             '  node [margin="0.18,0.10"];', '  ranksep=0.45; nodesep=0.35;']

    for name, label in QUERY_LABELS.items():
        text = label
        if name in facts:
            text += f"\\n{facts[name]}"
        lines.append(_node(name, text, state_of(name)))

    touched = done | ({current} if current else set())
    lines += [
        _edge("contextualize", "classify",
              active="classify" in touched),
        _edge("classify", "retrieve", "search",
              active=route == "search"),
        _edge("classify", "chat_reply", "chat",
              active=route == "chat"),
        _edge("retrieve", "grade_docs", active="grade_docs" in touched),
        _edge("grade_docs", "generate", "relevant",
              active="generate" in touched),
        _edge("grade_docs", "rewrite_query", "nothing kept",
              active="rewrite_query" in touched),
        _edge("grade_docs", "no_answer", "gave up",
              active="no_answer" in touched),
        # THE CYCLE - drawn dashed so the loop is visually obvious
        _edge("rewrite_query", "retrieve", "retry",
              active="rewrite_query" in touched, back=True),
        "}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Turning a LangGraph stream update into a one-line fact for the diagram
# ---------------------------------------------------------------------------
def fact_for(node: str, update: dict, question: str = "") -> str:
    """A short result string to print inside the node box.

    Deliberately terse - the box is small. The full detail goes to the step
    log beside the diagram and to logs/assistant.log.
    """
    if node == "contextualize":
        # `standalone_question` is always populated, so its mere presence
        # proves nothing. Compare it to what the user actually typed.
        q = update.get("standalone_question", "")
        if question and q.lower().strip() != question.lower().strip():
            return "rewritten"
        return "unchanged"
    if node == "classify":
        return f"-> {update.get('route', '')}"
    if node == "retrieve":
        n = len(update.get("documents") or [])
        return f"attempt {update.get('attempts', '?')}: {n} chunks"
    if node == "grade_docs":
        return f"kept {len(update.get('relevant_docs') or [])}"
    if node == "rewrite_query":
        return "new query"
    if node in ("generate", "chat_reply"):
        return "answered"
    if node == "no_answer":
        return "no answer"
    return ""


def next_node(node: str, update: dict, kept_total: int = 0,
              attempts: int = 0, max_attempts: int = 3) -> str | None:
    """Which node will run next, given the one that just finished.

    WHY THIS EXISTS: graph.stream() yields AFTER a node completes, so it can
    tell you what has finished but never what is running. Without this the
    diagram would only ever show green boxes and would go blank during the
    20 seconds a node is actually working - exactly when you want to know
    where you are.

    The graph's edges are fixed, so the next node is predictable except after
    grade_docs, where the router's choice depends on a verdict we have not
    seen yet. Returning None there is honest: we genuinely don't know.
    """
    if node == "contextualize":
        return "classify"
    if node == "classify":
        return "retrieve" if update.get("route") == "search" else "chat_reply"
    if node == "retrieve":
        return "grade_docs"
    if node == "rewrite_query":
        return "retrieve"
    if node == "grade_docs":
        # Mirror route_after_grading() in step9_memory.py. Worth doing rather
        # than returning None: generate is the LONGEST node, so leaving the
        # diagram unhighlighted here would blank it out for ~30 seconds -
        # precisely when the user most wants to know where they are.
        #
        # kept_total is the ACCUMULATED count across retry laps, not just
        # this lap's, because the reducer accumulates and the real router
        # reads the accumulated value.
        if kept_total:
            return "generate"
        if attempts >= max_attempts:
            return "no_answer"
        return "rewrite_query"
    return None                     # terminal node - nothing comes next
