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
    background: var(--background-color, #ffffff);
    border-left: 1px solid rgba(148,163,184,.30);
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
</style>
"""


def _esc(text) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def _card(node: str, state: str, detail: str = "",
          seconds: float | None = None) -> str:
    accent, glyph = STATES[state]
    time_html = (f'<div class="ftime">{seconds:.1f}s</div>'
                 if seconds is not None else "")
    detail_html = f'<div class="fdetail">{_esc(detail)}</div>' if detail else ""
    return (
        f'<div class="fnode is-{state}" style="--accent:{accent}">'
        f'<div class="fglyph">{glyph}</div>'
        f'<div class="fbody">'
        f'<div class="fname">{_esc(node)}</div>'
        f'<div class="fdesc">{_esc(NODE_DESC.get(node, ""))}</div>'
        f'{detail_html}</div>{time_html}</div>'
    )


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
def query_html(events: list[dict] | None = None, current: str | None = None,
               route: str | None = None) -> str:
    """Render one question's run.

    events:  nodes that have ALREADY finished, in order, each
             {"node", "detail", "seconds"}
    current: the node running right now - drawn pulsing at the end
    route:   "search" / "chat" once known - draws the branch pills

    Rendering the real execution ORDER rather than the static graph shape is
    what makes a retry legible: retrieve and grade_docs simply appear twice,
    with a loop marker between the laps.
    """
    events = events or []
    if not events and not current:
        return '<div class="fdesc">Nothing has run yet.</div>'

    parts = ['<div class="flow">']
    prev = None
    for i, ev in enumerate(events):
        node = ev["node"]
        if node == "retrieve" and prev == "rewrite_query":
            parts.append('<div class="floop">&#8635; retry &mdash; '
                         'back to retrieve</div>')
        elif i:
            parts.append('<div class="fconn"></div>')

        state = "failed" if node == "no_answer" else "done"
        parts.append(_card(node, state, ev.get("detail", ""),
                           ev.get("seconds")))

        if node == "classify" and route:
            search_cls = "taken" if route == "search" else ""
            chat_cls = "taken" if route == "chat" else ""
            parts.append(
                '<div class="fbranch">'
                f'<span class="fpill {search_cls}">search the documents</span>'
                f'<span class="fpill {chat_cls}">reply directly</span></div>'
            )
        prev = node

    if current:
        if events:
            parts.append('<div class="fconn"></div>')
        parts.append(_card(current, "running"))

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
