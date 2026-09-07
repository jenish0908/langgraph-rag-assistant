"""
The live pipeline view - a slide-in drawer showing exactly what the graph did.

Rendered as hand-written HTML/CSS rather than Graphviz. Streamlit draws DOT
with dagre-d3, which supports only a small subset of styling: no shadows, no
gradients, no animation, and little control over typography. Since we want a
running node to visibly pulse and a retry to read as a loop, plain HTML is
both prettier and simpler - and it removes a dependency.

Colours are semi-transparent so the same markup looks right on Streamlit's
light and dark themes without needing to know which one is active.
"""

# --- Palette ---------------------------------------------------------------
STATES = {
    #          accent (rgb)   glyph
    "done":    ("16,185,129", "&#10003;"),   # emerald, check
    "running": ("245,158,11", "&#9679;"),    # amber, pulses
    "pending": ("148,163,184", "&#9675;"),   # slate, hollow
    "skipped": ("148,163,184", "&#8212;"),   # faded, dash
    "failed":  ("239,68,68", "&#10005;"),    # red, cross
}

NODE_DESC = {
    "contextualize": "resolve follow-ups against history",
    "classify": "does this need the documents?",
    "chat_reply": "answer small talk directly",
    "retrieve": "find the k nearest chunks",
    "grade_docs": "do these chunks actually answer it?",
    "rewrite_query": "rephrase the question and retry",
    "generate": "write the answer with citations",
    "no_answer": "admit we don't know",
}

INGEST_STEPS = [
    ("load", "Load files", "read docs/ into Documents"),
    ("chunk", "Chunk", "split into overlapping pieces"),
    ("embed", "Embed", "text into 384-dim vectors"),
    ("store", "Store", "build the searchable index"),
]


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
def styles(drawer_open: bool) -> str:
    """All CSS for the drawer and the flow cards.

    The drawer is an ordinary Streamlit container pinned to the right with
    position:fixed. `st.container(key="steps_drawer")` renders with the class
    `st-key-steps_drawer`, which is the supported hook for styling a
    container - no custom component required.

    Open/closed is a CSS transform rather than rendering/not-rendering the
    content, so the panel slides instead of blinking, and its contents stay
    alive underneath while a question is running.
    """
    shift = "0" if drawer_open else "105%"
    shadow = "-18px 0 48px rgba(0,0,0,.28)" if drawer_open else "none"
    return f"""
<style>
/* ---------- the slide-in drawer ---------- */
.st-key-steps_drawer {{
    position: fixed;
    top: 0; right: 0;
    width: min(460px, 92vw);
    height: 100vh;
    z-index: 9990;
    padding: 3.2rem 1.15rem 1.5rem 1.15rem;
    overflow-y: auto;
    /* Explicit black rather than var(--background-color): the drawer is
       position:fixed and sits outside the normal flow, so it does not
       inherit the canvas colour reliably. */
    background: #000000;
    border-left: 1px solid rgba(148,163,184,.22);
    box-shadow: {shadow};
    transform: translateX({shift});
    transition: transform .32s cubic-bezier(.4,0,.2,1), box-shadow .32s ease;
}}
.st-key-steps_drawer::-webkit-scrollbar {{ width: 8px; }}
.st-key-steps_drawer::-webkit-scrollbar-thumb {{
    background: rgba(148,163,184,.4); border-radius: 4px;
}}

/* ---------- flow cards ---------- */
.flow {{ display: flex; flex-direction: column; gap: 0; }}

.fnode {{
    display: flex; align-items: flex-start; gap: .65rem;
    padding: .6rem .75rem;
    border: 1px solid rgba(var(--accent), .34);
    background: rgba(var(--accent), .075);
    border-radius: 11px;
}}
.fnode.is-running {{
    border-color: rgba(var(--accent), .75);
    animation: fpulse 1.5s ease-in-out infinite;
}}
.fnode.is-skipped, .fnode.is-pending {{ opacity: .5; border-style: dashed; }}

@keyframes fpulse {{
    0%, 100% {{ box-shadow: 0 0 0 0 rgba(var(--accent), .40); }}
    50%      {{ box-shadow: 0 0 0 7px rgba(var(--accent), 0); }}
}}

.fglyph {{
    flex: 0 0 22px; height: 22px; line-height: 22px; text-align: center;
    border-radius: 50%; font-size: 11px; font-weight: 700;
    color: rgb(var(--accent));
    background: rgba(var(--accent), .16);
}}
.fbody {{ flex: 1 1 auto; min-width: 0; }}
.fname {{
    font-weight: 650; font-size: .84rem; letter-spacing: .01em;
    font-family: ui-monospace, "Cascadia Code", Consolas, monospace;
}}
.fdesc {{ font-size: .71rem; opacity: .62; margin-top: .05rem; }}
.fdetail {{
    font-size: .71rem; margin-top: .32rem;
    font-family: ui-monospace, "Cascadia Code", Consolas, monospace;
    background: rgba(148,163,184,.13);
    padding: .2rem .42rem; border-radius: 5px;
    word-break: break-word;
}}
.ftime {{
    flex: 0 0 auto; font-size: .68rem; opacity: .55;
    font-variant-numeric: tabular-nums; padding-top: .15rem;
}}

/* ---------- connectors ---------- */
.fconn {{
    width: 2px; height: 14px; margin-left: 1.42rem;
    background: linear-gradient(rgba(148,163,184,.55), rgba(148,163,184,.18));
}}
.floop {{
    display: flex; align-items: center; gap: .4rem;
    margin: .25rem 0 .25rem .95rem;
    font-size: .68rem; font-weight: 650;
    color: rgb(245,158,11);
}}
.fbranch {{
    display: flex; gap: .4rem; margin: .35rem 0 .35rem 1.9rem;
    font-size: .66rem; flex-wrap: wrap;
}}
.fpill {{
    padding: .12rem .5rem; border-radius: 999px;
    border: 1px solid rgba(148,163,184,.4); opacity: .45;
}}
.fpill.taken {{
    border-color: rgba(16,185,129,.6);
    background: rgba(16,185,129,.13);
    color: rgb(16,185,129); opacity: 1; font-weight: 650;
}}

/* ---------- ingestion strip ---------- */
.istrip {{ display: flex; gap: .45rem; align-items: stretch; }}
.istep {{
    flex: 1 1 0; min-width: 0;
    border: 1px solid rgba(var(--accent), .34);
    background: rgba(var(--accent), .075);
    border-radius: 10px; padding: .5rem .6rem;
}}
.istep.is-running {{ animation: fpulse 1.5s ease-in-out infinite; }}
.istep.is-pending {{ opacity: .5; border-style: dashed; }}
.ihead {{ font-size: .74rem; font-weight: 650; }}
.idesc {{ font-size: .66rem; opacity: .6; margin-top: .1rem; }}
.ifact {{
    font-size: .67rem; margin-top: .3rem; font-weight: 600;
    color: rgb(var(--accent));
    font-family: ui-monospace, Consolas, monospace;
}}
.iarrow {{ align-self: center; opacity: .35; font-size: .8rem; }}

/* ---------- expandable step (<details>) ---------- */
.fstep {{ border: 0; }}
.fstep > summary {{ cursor: pointer; list-style: none; }}
.fstep > summary::-webkit-details-marker {{ display: none; }}
.fstep > summary:hover {{ border-color: rgba(var(--accent), .68); }}
.fchev {{
    flex: 0 0 auto; opacity: .5; font-size: .95rem; padding-top: .05rem;
    transition: transform .18s ease;
}}
.fstep[open] > summary .fchev {{ transform: rotate(90deg); }}

.fout {{
    margin: .3rem 0 .1rem 1.05rem;
    padding: .6rem .75rem;
    border-left: 2px solid rgba(var(--accent), .45);
    background: rgba(148,163,184,.07);
    border-radius: 0 8px 8px 0;
    font-size: .71rem; line-height: 1.45;
}}
.fout b {{ display: block; margin: .45rem 0 .25rem; font-size: .69rem;
           text-transform: uppercase; letter-spacing: .04em; opacity: .65; }}
.fout b:first-child {{ margin-top: 0; }}
.fkv {{ display: flex; gap: .5rem; margin: .16rem 0; }}
.fk {{ flex: 0 0 84px; opacity: .55; }}
.fv {{ flex: 1 1 auto; word-break: break-word; }}
.fraw {{
    background: rgba(148,163,184,.16); padding: .28rem .45rem;
    border-radius: 5px; font-family: ui-monospace, Consolas, monospace;
    word-break: break-word;
}}
.fnote {{ margin-top: .45rem; opacity: .5; font-style: italic;
          font-size: .67rem; }}
.fchunk {{
    border-left: 2px solid rgba(148,163,184,.3);
    padding-left: .5rem; margin: .38rem 0;
}}
.fchead {{ font-family: ui-monospace, Consolas, monospace; font-size: .68rem; }}
.fctext {{ opacity: .62; margin-top: .12rem; }}
.fdim {{ opacity: .4; }}
.fscore {{ color: rgb(16,185,129); font-weight: 700; }}
.fverdict {{ margin: .16rem 0; font-family: ui-monospace, Consolas, monospace;
             font-size: .68rem; }}
.fkeep {{ color: rgb(16,185,129); font-weight: 700; }}
.fdrop {{ color: rgb(239,68,68); font-weight: 700; opacity: .85; }}
.fanswer {{
    background: rgba(16,185,129,.09); border-radius: 6px;
    padding: .4rem .5rem; line-height: 1.5;
}}
.frun {{ margin: .5rem 0 .2rem; padding-top: .35rem;
         border-top: 1px dashed rgba(148,163,184,.25); }}
.frun:first-child {{ border-top: 0; padding-top: 0; margin-top: 0; }}
.frunhead {{ font-size: .66rem; font-weight: 700; opacity: .55;
             text-transform: uppercase; letter-spacing: .04em; }}

/* ---------- graph structure ---------- */
.gwrap {{ display: flex; flex-direction: column; gap: .1rem; }}
.grow {{ display: flex; align-items: stretch; }}
.grail {{
    flex: 0 0 auto; width: 0; border-left: 2px solid rgba(148,163,184,.22);
    margin-right: .55rem;
}}
.grow.d0 > .grail {{ display: none; }}
.grow.d1 {{ padding-left: .45rem; }}
.grow.d2 {{ padding-left: 1.35rem; }}
.gbody {{ flex: 1 1 auto; min-width: 0; }}
.gedge {{
    font-size: .62rem; letter-spacing: .03em; text-transform: uppercase;
    opacity: .5; margin: .3rem 0 .18rem .1rem;
}}
.gedge.taken {{ color: rgb(16,185,129); opacity: .95; font-weight: 700; }}
.gcount {{
    display: inline-block; margin-left: .35rem; padding: 0 .32rem;
    border-radius: 999px; font-size: .6rem; font-weight: 700;
    background: rgba(var(--accent), .2); color: rgb(var(--accent));
}}
.gloop {{
    display: flex; align-items: center; gap: .35rem;
    margin: .25rem 0 .1rem 1.35rem;
    font-size: .65rem; font-weight: 700; color: rgb(245,158,11);
}}
.gloop.idle {{ color: rgba(148,163,184,.55); font-weight: 500; }}
</style>
"""


def _esc(text) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def _card(node: str, state: str, detail: str = "",
          seconds: float | None = None, output: str = "") -> str:
    """One step.

    With `output`, the card becomes a native <details> disclosure: clicking it
    reveals what that step actually produced. <details> is used rather than
    st.expander deliberately - it toggles in the browser with no Streamlit
    rerun, so opening a step mid-question cannot disturb the streaming loop.
    """
    accent, glyph = STATES[state]
    time_html = (f'<div class="ftime">{seconds:.1f}s</div>'
                 if seconds is not None else "")
    detail_html = f'<div class="fdetail">{_esc(detail)}</div>' if detail else ""
    chevron = '<div class="fchev">&#8250;</div>' if output else ""

    inner = (
        f'<div class="fglyph">{glyph}</div>'
        f'<div class="fbody">'
        f'<div class="fname">{_esc(node)}</div>'
        f'<div class="fdesc">{_esc(NODE_DESC.get(node, ""))}</div>'
        f'{detail_html}</div>{time_html}{chevron}'
    )

    if not output:
        return (f'<div class="fnode is-{state}" style="--accent:{accent}">'
                f'{inner}</div>')

    return (
        f'<details class="fstep" style="--accent:{accent}">'
        f'<summary class="fnode is-{state}">{inner}</summary>'
        f'<div class="fout">{output}</div>'
        f'</details>'
    )


# ---------------------------------------------------------------------------
# WHAT EACH STEP PRODUCED  (the expanded view)
# ---------------------------------------------------------------------------
def _kv(key: str, value) -> str:
    return (f'<div class="fkv"><div class="fk">{_esc(key)}</div>'
            f'<div class="fv">{_esc(value)}</div></div>')


def _raw(text: str) -> str:
    return f'<div class="fraw">{_esc(text)}</div>'


def output_html(node: str, update: dict, tev: dict | None = None,
                question: str = "") -> str:
    """Build the expanded 'what did this step produce' panel.

    Two sources, because neither alone is enough:
      * `update` - what the node returned into graph state (Documents with
        their text, the answer)
      * `tev` - the matching trace_log event, which carries the things that
        never reach state: similarity scores, the grader's RAW reply, token
        counts
    """
    tev = tev or {}
    p: list[str] = []

    if node == "contextualize":
        p.append(_kv("you typed", question))
        p.append(_kv("searched as", update.get("standalone_question", "")))
        p.append(_kv("history", f"{tev.get('history_messages', 0)} messages"))
        if not tev.get("rewritten"):
            p.append('<div class="fnote">Already self-contained - no rewrite '
                     'needed.</div>')

    elif node == "classify":
        p.append("<b>Raw model reply</b>")
        p.append(_raw(tev.get("raw", "?")))
        p.append(_kv("decision", update.get("route", "")))
        p.append('<div class="fnote">"search" runs retrieval; "chat" skips '
                 'the documents entirely.</div>')

    elif node == "retrieve":
        p.append(_kv("query used", tev.get("query", "")))
        scores = {f"{r['source']}#{r['chunk_id']}": r["score"]
                  for r in tev.get("results", [])}
        p.append(f"<b>{len(update.get('documents', []))} nearest chunks</b>")
        for d in update.get("documents", []):
            label = f"{d.metadata['source']}#{d.metadata['chunk_id']}"
            score = scores.get(label)
            score_html = (f'<span class="fscore">{score:.3f}</span>'
                          if score is not None else "")
            body = d.page_content.strip().replace("\n", " ")[:320]
            p.append(f'<div class="fchunk"><div class="fchead">{_esc(label)} '
                     f'{score_html} <span class="fdim">'
                     f'{len(d.page_content)} chars</span></div>'
                     f'<div class="fctext">{_esc(body)}...</div></div>')
        p.append('<div class="fnote">Score is cosine similarity: 1.0 is '
                 'identical meaning, ~0.5 is weakly related.</div>')

    elif node == "grade_docs":
        p.append("<b>Raw model reply</b>")
        p.append(_raw(tev.get("raw", "?")))
        p.append(f"<b>Verdicts &mdash; kept {tev.get('kept', '?')}"
                 f"/{tev.get('total', '?')}</b>")
        for v in tev.get("verdicts", []):
            keep = v["keep"]
            cls = "fkeep" if keep else "fdrop"
            word = "KEEP" if keep else "drop"
            p.append(f'<div class="fverdict"><span class="{cls}">{word}</span>'
                     f' {_esc(v["source"])}#{v["chunk_id"]}</div>')
        p.append('<div class="fnote">Dropped chunks are never shown to the '
                 'answering model - that is what stops it inventing an '
                 'answer from near-miss text.</div>')

    elif node == "rewrite_query":
        p.append(_kv("failed query", tev.get("previous_query", "")))
        p.append(_kv("new query", update.get("question", "")))
        tried = tev.get("tried", [])
        if tried:
            p.append("<b>Already tried</b>")
            for t in tried:
                p.append(f'<div class="fchunk">{_esc(t)}</div>')
        p.append('<div class="fnote">The graph now loops back to retrieve '
                 'with this new wording.</div>')

    elif node in ("generate", "chat_reply"):
        used = tev.get("chunks_used")
        if used:
            p.append(_kv("built from", ", ".join(used)))
        if tev.get("prompt_tokens"):
            p.append(_kv("tokens", f"{tev['prompt_tokens']} in / "
                                   f"{tev.get('output_tokens', '?')} out"))
        p.append("<b>Answer</b>")
        p.append(f'<div class="fanswer">{_esc(update.get("answer", ""))}</div>')

    elif node == "no_answer":
        p.append(_kv("attempts", tev.get("attempts", "?")))
        p.append('<div class="fnote">Every retrieved chunk was rejected by '
                 'the grader on every attempt, so the model was never asked '
                 'to answer. It cannot hallucinate what it was not asked.'
                 '</div>')

    return "".join(p)


# ---------------------------------------------------------------------------
# INGESTION STRIP
# ---------------------------------------------------------------------------
def ingestion_html(current: str | None = None, done: set | None = None,
                   facts: dict | None = None) -> str:
    done, facts = done or set(), facts or {}
    parts = ['<div class="istrip">']
    for i, (key, title, desc) in enumerate(INGEST_STEPS):
        state = ("running" if key == current
                 else "done" if key in done else "pending")
        accent, glyph = STATES[state]
        fact = facts.get(key, "")
        parts.append(
            f'<div class="istep is-{state}" style="--accent:{accent}">'
            f'<div class="ihead">{glyph} {i + 1}. {_esc(title)}</div>'
            f'<div class="idesc">{_esc(desc)}</div>'
            + (f'<div class="ifact">{_esc(fact)}</div>' if fact else "")
            + "</div>"
        )
        if i < len(INGEST_STEPS) - 1:
            parts.append('<div class="iarrow">&#8594;</div>')
    parts.append("</div>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# QUERY FLOW
# ---------------------------------------------------------------------------
# The STATIC shape of the graph, in reading order.
#   (node, depth, edge label that leads to it)
# Depth expresses the branch structure: depth 1 hangs off classify's decision,
# depth 2 off grade_docs'. This list is the single source of truth for the
# picture - it mirrors build_graph() in step9_memory.py.
GRAPH_LAYOUT = [
    ("contextualize", 0, None),
    ("classify", 0, None),
    ("chat_reply", 1, "chat"),
    ("retrieve", 1, "search"),
    ("grade_docs", 1, None),
    ("generate", 2, "relevant"),
    ("rewrite_query", 2, "nothing kept"),
    ("no_answer", 2, "attempts exhausted"),
]

# Which nodes belong to which branch out of classify.
SEARCH_BRANCH = {"retrieve", "grade_docs", "generate", "rewrite_query",
                 "no_answer"}


def query_html(runs: dict | None = None, current: str | None = None,
               route: str | None = None, finished: bool = False) -> str:
    """Draw the WHOLE graph, with the path actually taken highlighted.

    runs:     node -> list of runs, each {"detail", "seconds", "output"}.
              A list because retrieve and grade_docs execute more than once
              when the retry cycle fires.
    current:  the node executing right now (pulses).
    route:    "search" / "chat" once classify has decided - the branch not
              taken is dimmed.
    finished: once the run is over, nodes that were never reached become
              "skipped" rather than "pending", so a completed answer shows a
              settled picture instead of one that looks still in progress.

    Every node is ALWAYS drawn. That is the point: you can see the decisions
    that were available, not just the ones that happened.
    """
    runs = runs or {}

    def state_of(node: str) -> str:
        if node == current:
            return "running"
        if node in runs:
            return "failed" if node == "no_answer" else "done"
        # Branch not taken - dim it as soon as the router has decided.
        if route == "chat" and node in SEARCH_BRANCH:
            return "skipped"
        if route == "search" and node == "chat_reply":
            return "skipped"
        return "skipped" if finished else "pending"

    parts = ['<div class="gwrap">']

    for node, depth, edge in GRAPH_LAYOUT:
        state = state_of(node)
        node_runs = runs.get(node, [])

        if edge:
            taken = "taken" if node_runs or node == current else ""
            parts.append(f'<div class="gedge {taken}">&#8627; {_esc(edge)}</div>')

        # Multiple runs are folded into one card with a ×N badge; the expanded
        # panel then lists each attempt separately.
        if node_runs:
            last = node_runs[-1]
            detail = last.get("detail", "")
            seconds = sum(r.get("seconds", 0) or 0 for r in node_runs)
            if len(node_runs) > 1:
                output = "".join(
                    f'<div class="frun"><div class="frunhead">'
                    f'attempt {i}</div>{r.get("output", "")}</div>'
                    for i, r in enumerate(node_runs, start=1))
            else:
                output = last.get("output", "")
        else:
            detail, seconds, output = "", None, ""

        accent = STATES[state][0]
        count = (f'<span class="gcount" style="--accent:{accent}">'
                 f'&#215;{len(node_runs)}</span>' if len(node_runs) > 1 else "")

        card = _card(node, state, detail, seconds, output)
        if count:                       # slot the badge in beside the name
            card = card.replace('</div><div class="fdesc">',
                                f'{count}</div><div class="fdesc">', 1)

        parts.append(f'<div class="grow d{depth}"><div class="grail"></div>'
                     f'<div class="gbody">{card}</div></div>')

        # The cycle, drawn under the node it returns from.
        if node == "rewrite_query":
            fired = len(runs.get("retrieve", [])) > 1
            cls = "" if fired else "idle"
            label = ("&#8635; loops back to retrieve" if fired
                     else "&#8635; would loop back to retrieve")
            parts.append(f'<div class="gloop {cls}">{label}</div>')

    parts.append("</div>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Interpreting a LangGraph stream update
# ---------------------------------------------------------------------------
def fact_for(node: str, update: dict, question: str = "") -> str:
    """A short result string for the node card."""
    if node == "contextualize":
        # `standalone_question` is always populated, so its mere presence
        # proves nothing. Compare it to what the user actually typed.
        q = update.get("standalone_question", "")
        if question and q.lower().strip() != question.lower().strip():
            return f"rewritten: {q}"
        return "already self-contained"
    if node == "classify":
        return f"route: {update.get('route', '')}"
    if node == "retrieve":
        n = len(update.get("documents") or [])
        return f"attempt {update.get('attempts', '?')} - {n} chunks"
    if node == "grade_docs":
        return f"kept {len(update.get('relevant_docs') or [])}"
    if node == "rewrite_query":
        return update.get("question", "new query")
    if node in ("generate", "chat_reply"):
        return ""
    if node == "no_answer":
        return "nothing relevant survived grading"
    return ""


def next_node(node: str, update: dict, kept_total: int = 0,
              attempts: int = 0, max_attempts: int = 3) -> str | None:
    """Which node will run next, given the one that just finished.

    WHY THIS EXISTS: graph.stream() yields AFTER a node completes, so it can
    tell you what has finished but never what is running. Without this the
    view would show only completed steps and would sit still through the
    20-30 seconds a node is actually working - exactly when you want to know
    where you are.
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
        # Mirror route_after_grading() in step9_memory.py. kept_total is the
        # ACCUMULATED count across retry laps, because the reducer accumulates
        # and the real router reads the accumulated value.
        if kept_total:
            return "generate"
        if attempts >= max_attempts:
            return "no_answer"
        return "rewrite_query"
    return None                     # terminal node - nothing comes next
