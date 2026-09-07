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
def styles() -> str:
    """All CSS for the app.

    The drawer is a single div we own, rendered inside one st.markdown block
    and pinned to the right. It is ALWAYS open, so the main content and the
    chat input are both offset to the left by its width - otherwise they sit
    underneath it. Streamlit's own containers are the ones being offset:
    .stMainBlockContainer for the page, .stBottomBlockContainer for the
    fixed chat input.

    Below 820px the panel is hidden and the chat takes the full width -
    there is not room for both on a phone.
    """
    return """
<style>
/* Panel width in one place - the offsets below are derived from it. */
:root { --dw: 430px; }

/* ---------- the always-open right panel ----------
   This is OUR div inside a single st.markdown block, not a Streamlit
   container. An earlier version styled st.container(key=...) - the class
   name was right, but its children are Streamlit's own nested block
   wrappers, and re-parenting those into a fixed panel left the panel
   painting its background with no visible content. Owning the whole
   subtree removes that entire class of problem. */
.ragdrawer {
    position: fixed;
    top: 0; right: 0;
    width: var(--dw);
    height: 100vh;
    z-index: 999990;
    padding: 1.1rem 1.15rem 2rem 1.15rem;
    overflow-y: auto;
    background: #050506;
    color: #e8e8ea;
    border-left: 1px solid rgba(148,163,184,.22);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
}
.ragdrawer::-webkit-scrollbar { width: 8px; }
.ragdrawer::-webkit-scrollbar-thumb {
    background: rgba(148,163,184,.4); border-radius: 4px;
}

/* Keep the app clear of the panel. */
.stMainBlockContainer { padding-right: calc(var(--dw) + 2rem) !important; }
.stBottomBlockContainer { padding-right: calc(var(--dw) + 2rem) !important; }

@media (max-width: 1200px) { :root { --dw: 360px; } }

/* Below this there is no room for both; the chat wins. */
@media (max-width: 820px) {
    .ragdrawer { display: none; }
    .stMainBlockContainer, .stBottomBlockContainer {
        padding-right: 1rem !important;
    }
}

.dtitle { font-size: .95rem; font-weight: 700; margin: 0 0 .1rem; }
.dsub { font-size: .7rem; opacity: .55; margin: 0 0 .7rem; }
.dsec {
    font-size: .66rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: .06em; opacity: .5;
    margin: 1.1rem 0 .4rem; padding-top: .8rem;
    border-top: 1px solid rgba(148,163,184,.16);
}
.dsec:first-of-type { border-top: 0; padding-top: 0; margin-top: .2rem; }
.dfile {
    font-family: ui-monospace, Consolas, monospace; font-size: .72rem;
    color: rgb(16,185,129); background: rgba(16,185,129,.09);
    border-radius: 6px; padding: .22rem .5rem; margin-bottom: .25rem;
}

/* ---------- flow cards ---------- */
.flow { display: flex; flex-direction: column; gap: 0; }

.fnode {
    display: flex; align-items: flex-start; gap: .65rem;
    padding: .6rem .75rem;
    border: 1px solid rgba(var(--accent), .34);
    background: rgba(var(--accent), .075);
    border-radius: 11px;
}
.fnode.is-running {
    border-color: rgba(var(--accent), .75);
    animation: fpulse 1.5s ease-in-out infinite;
}
.fnode.is-skipped, .fnode.is-pending { opacity: .5; border-style: dashed; }

@keyframes fpulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(var(--accent), .40); }
    50%      { box-shadow: 0 0 0 7px rgba(var(--accent), 0); }
}

.fglyph {
    flex: 0 0 22px; height: 22px; line-height: 22px; text-align: center;
    border-radius: 50%; font-size: 11px; font-weight: 700;
    color: rgb(var(--accent));
    background: rgba(var(--accent), .16);
}
.fbody { flex: 1 1 auto; min-width: 0; }
.fname {
    font-weight: 650; font-size: .84rem; letter-spacing: .01em;
    font-family: ui-monospace, "Cascadia Code", Consolas, monospace;
}
.fdesc { font-size: .71rem; opacity: .62; margin-top: .05rem; }
.fdetail {
    font-size: .71rem; margin-top: .32rem;
    font-family: ui-monospace, "Cascadia Code", Consolas, monospace;
    background: rgba(148,163,184,.13);
    padding: .2rem .42rem; border-radius: 5px;
    word-break: break-word;
}
.ftime {
    flex: 0 0 auto; font-size: .68rem; opacity: .55;
    font-variant-numeric: tabular-nums; padding-top: .15rem;
}

/* ---------- connectors ---------- */
.fconn {
    width: 2px; height: 14px; margin-left: 1.42rem;
    background: linear-gradient(rgba(148,163,184,.55), rgba(148,163,184,.18));
}
.floop {
    display: flex; align-items: center; gap: .4rem;
    margin: .25rem 0 .25rem .95rem;
    font-size: .68rem; font-weight: 650;
    color: rgb(245,158,11);
}
.fbranch {
    display: flex; gap: .4rem; margin: .35rem 0 .35rem 1.9rem;
    font-size: .66rem; flex-wrap: wrap;
}
.fpill {
    padding: .12rem .5rem; border-radius: 999px;
    border: 1px solid rgba(148,163,184,.4); opacity: .45;
}
.fpill.taken {
    border-color: rgba(16,185,129,.6);
    background: rgba(16,185,129,.13);
    color: rgb(16,185,129); opacity: 1; font-weight: 650;
}

/* ---------- ingestion strip ---------- */
.istrip { display: flex; gap: .45rem; align-items: stretch; }
.istep {
    flex: 1 1 0; min-width: 0;
    border: 1px solid rgba(var(--accent), .34);
    background: rgba(var(--accent), .075);
    border-radius: 10px; padding: .5rem .6rem;
}
.istep.is-running { animation: fpulse 1.5s ease-in-out infinite; }
.istep.is-pending { opacity: .5; border-style: dashed; }
.ihead { font-size: .74rem; font-weight: 650; }
.idesc { font-size: .66rem; opacity: .6; margin-top: .1rem; }
.ifact {
    font-size: .67rem; margin-top: .3rem; font-weight: 600;
    color: rgb(var(--accent));
    font-family: ui-monospace, Consolas, monospace;
}
.iarrow { align-self: center; opacity: .35; font-size: .8rem; }

/* ---------- expandable step (<details>) ---------- */
.fstep { border: 0; }
.fstep > summary { cursor: pointer; list-style: none; }
.fstep > summary::-webkit-details-marker { display: none; }
.fstep > summary:hover { border-color: rgba(var(--accent), .68); }
.fchev {
    flex: 0 0 auto; opacity: .5; font-size: .95rem; padding-top: .05rem;
    transition: transform .18s ease;
}
.fstep[open] > summary .fchev { transform: rotate(90deg); }

.fout {
    margin: .3rem 0 .1rem 1.05rem;
    padding: .6rem .75rem;
    border-left: 2px solid rgba(var(--accent), .45);
    background: rgba(148,163,184,.07);
    border-radius: 0 8px 8px 0;
    font-size: .71rem; line-height: 1.45;
}
.fout b { display: block; margin: .45rem 0 .25rem; font-size: .69rem;
           text-transform: uppercase; letter-spacing: .04em; opacity: .65; }
.fout b:first-child { margin-top: 0; }
.fkv { display: flex; gap: .5rem; margin: .16rem 0; }
.fk { flex: 0 0 84px; opacity: .55; }
.fv { flex: 1 1 auto; word-break: break-word; }
.fraw {
    background: rgba(148,163,184,.16); padding: .28rem .45rem;
    border-radius: 5px; font-family: ui-monospace, Consolas, monospace;
    word-break: break-word;
}
.fnote { margin-top: .45rem; opacity: .5; font-style: italic;
          font-size: .67rem; }
.fchunk {
    border-left: 2px solid rgba(148,163,184,.3);
    padding-left: .5rem; margin: .38rem 0;
}
.fchead { font-family: ui-monospace, Consolas, monospace; font-size: .68rem; }
.fctext { opacity: .62; margin-top: .12rem; }
.fdim { opacity: .4; }
.fscore { color: rgb(16,185,129); font-weight: 700; }
.fverdict { margin: .16rem 0; font-family: ui-monospace, Consolas, monospace;
             font-size: .68rem; }
.fkeep { color: rgb(16,185,129); font-weight: 700; }
.fdrop { color: rgb(239,68,68); font-weight: 700; opacity: .85; }
.fanswer {
    background: rgba(16,185,129,.09); border-radius: 6px;
    padding: .4rem .5rem; line-height: 1.5;
}
.frun { margin: .5rem 0 .2rem; padding-top: .35rem;
         border-top: 1px dashed rgba(148,163,184,.25); }
.frun:first-child { border-top: 0; padding-top: 0; margin-top: 0; }
.frunhead { font-size: .66rem; font-weight: 700; opacity: .55;
             text-transform: uppercase; letter-spacing: .04em; }

/* ---------- graph structure ---------- */
.gwrap { display: flex; flex-direction: column; gap: .1rem; }
.grow { display: flex; align-items: stretch; }
.grail {
    flex: 0 0 auto; width: 0; border-left: 2px solid rgba(148,163,184,.22);
    margin-right: .55rem;
}
.grow.d0 > .grail { display: none; }
.grow.d1 { padding-left: .45rem; }
.grow.d2 { padding-left: 1.35rem; }
.gbody { flex: 1 1 auto; min-width: 0; }
.gedge {
    font-size: .62rem; letter-spacing: .03em; text-transform: uppercase;
    opacity: .5; margin: .3rem 0 .18rem .1rem;
}
.gedge.taken { color: rgb(16,185,129); opacity: .95; font-weight: 700; }
.gcount {
    display: inline-block; margin-left: .35rem; padding: 0 .32rem;
    border-radius: 999px; font-size: .6rem; font-weight: 700;
    background: rgba(var(--accent), .2); color: rgb(var(--accent));
}
.gloop {
    display: flex; align-items: center; gap: .35rem;
    margin: .25rem 0 .1rem 1.35rem;
    font-size: .65rem; font-weight: 700; color: rgb(245,158,11);
}
.gloop.idle { color: rgba(148,163,184,.55); font-weight: 500; }

/* ---------- the SVG diagram ---------- */
.gsvg { width: 100%; height: auto; display: block; margin: .2rem 0 .5rem; }
.gsvg text { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
.gpulse { animation: svgpulse 1.5s ease-in-out infinite; }
@keyframes svgpulse {
    0%, 100% { filter: drop-shadow(0 0 0 rgba(245,158,11,.55)); }
    50%      { filter: drop-shadow(0 0 7px rgba(245,158,11,.85)); }
}
.gsteps { margin-top: .3rem; }
.gsteps > .fstep { margin-bottom: .25rem; }
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
# THE GRAPH DIAGRAM (SVG)
# ---------------------------------------------------------------------------
# Hand-placed coordinates. The graph's shape never changes, so a fixed layout
# beats an auto-layout engine: it is stable frame to frame (nodes never jump
# as state updates), it fits the drawer's width exactly, and it needs no
# dependency. viewBox units; the SVG scales to whatever width it is given.
NODE_XY = {
    "contextualize": (240, 26),
    "classify":      (240, 94),
    "retrieve":      (170, 168),
    "chat_reply":    (392, 168),
    "grade_docs":    (170, 240),
    "generate":      (62, 322),
    "rewrite_query": (182, 322),
    "no_answer":     (312, 322),
}
NODE_W, NODE_H = 100, 34
SVG_W, SVG_H = 480, 380

# (from, to, path, label, label x/y)
EDGES = [
    ("contextualize", "classify", "M240,43 L240,77", None, None),
    ("classify", "retrieve",
     "M232,111 C210,132 190,140 172,151", "search", (176, 132)),
    ("classify", "chat_reply",
     "M256,111 C300,134 350,140 384,151", "chat", (316, 132)),
    ("retrieve", "grade_docs", "M170,185 L170,223", None, None),
    ("grade_docs", "generate",
     "M150,257 C120,278 86,292 66,305", "relevant", (78, 280)),
    ("grade_docs", "rewrite_query",
     "M174,257 C178,278 180,292 182,305", "none kept", (216, 283)),
    ("grade_docs", "no_answer",
     "M194,257 C240,280 286,296 306,305", "exhausted", (290, 275)),
]

# The cycle. Routed up the corridor between grade_docs (ends x=220) and
# chat_reply (starts x=342) so it crosses nothing.
LOOP_PATH = "M232,322 C268,314 272,236 254,200 C244,180 226,172 222,169"


def _svg_node(node: str, state: str, detail: str, count: int) -> str:
    accent = STATES[state][0]
    x, y = NODE_XY[node]
    left, top = x - NODE_W / 2, y - NODE_H / 2
    dash = ' stroke-dasharray="4 3"' if state in ("pending", "skipped") else ""
    op = ' opacity="0.45"' if state in ("pending", "skipped") else ""
    pulse = ' class="gpulse"' if state == "running" else ""
    badge = ""
    if count > 1:
        badge = (f'<circle cx="{left + NODE_W - 7}" cy="{top + 7}" r="8" '
                 f'fill="rgb({accent})"/>'
                 f'<text x="{left + NODE_W - 7}" y="{top + 10}" '
                 f'text-anchor="middle" font-size="9" font-weight="700" '
                 f'fill="#000">{count}</text>')
    sub = (f'<text x="{x}" y="{y + 11}" text-anchor="middle" font-size="8.5" '
           f'fill="rgb({accent})" opacity=".85">{_esc(detail[:22])}</text>'
           if detail else "")
    dy = -2 if detail else 4
    return (
        f'<g{op}>'
        f'<rect x="{left}" y="{top}" width="{NODE_W}" height="{NODE_H}" '
        f'rx="9" fill="rgba({accent},0.13)" stroke="rgb({accent})" '
        f'stroke-width="{2 if state == "running" else 1.3}"{dash}{pulse}/>'
        f'<text x="{x}" y="{y + dy}" text-anchor="middle" font-size="10.5" '
        f'font-weight="600" fill="rgb({accent})" '
        f'font-family="ui-monospace,Consolas,monospace">{_esc(node)}</text>'
        f'{sub}{badge}</g>'
    )


def graph_svg(runs: dict | None = None, current: str | None = None,
              route: str | None = None, finished: bool = False) -> str:
    """Draw the whole graph as a node-link diagram, path highlighted."""
    runs = runs or {}

    def state_of(node: str) -> str:
        if node == current:
            return "running"
        if node in runs:
            return "failed" if node == "no_answer" else "done"
        if route == "chat" and node in SEARCH_BRANCH:
            return "skipped"
        if route == "search" and node == "chat_reply":
            return "skipped"
        return "skipped" if finished else "pending"

    def edge_live(a: str, b: str) -> bool:
        """An edge is lit once its destination has been reached."""
        return b in runs or b == current

    parts = [
        f'<svg viewBox="0 0 {SVG_W} {SVG_H}" class="gsvg" '
        f'xmlns="http://www.w3.org/2000/svg">',
        '<defs>'
        '<marker id="ah" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="6" '
        'markerHeight="6" orient="auto-start-reverse">'
        '<path d="M0,0 L8,4 L0,8 z" fill="rgb(16,185,129)"/></marker>'
        '<marker id="ahd" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="6" '
        'markerHeight="6" orient="auto-start-reverse">'
        '<path d="M0,0 L8,4 L0,8 z" fill="rgba(148,163,184,.55)"/></marker>'
        '<marker id="ahl" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="6" '
        'markerHeight="6" orient="auto-start-reverse">'
        '<path d="M0,0 L8,4 L0,8 z" fill="rgb(245,158,11)"/></marker>'
        '</defs>',
    ]

    # --- edges first, so nodes paint over their ends ---
    for a, b, path, label, lxy in EDGES:
        live = edge_live(a, b)
        colour = "rgb(16,185,129)" if live else "rgba(148,163,184,.32)"
        width = 2 if live else 1.2
        marker = "ah" if live else "ahd"
        parts.append(f'<path d="{path}" fill="none" stroke="{colour}" '
                     f'stroke-width="{width}" marker-end="url(#{marker})"/>')
        if label and lxy:
            lx, ly = lxy
            lc = "rgb(16,185,129)" if live else "rgba(148,163,184,.6)"
            parts.append(f'<text x="{lx}" y="{ly}" text-anchor="middle" '
                         f'font-size="8" fill="{lc}" '
                         f'font-weight="{"700" if live else "400"}">'
                         f'{_esc(label)}</text>')

    # --- the cycle ---
    fired = len(runs.get("retrieve", [])) > 1
    lc = "rgb(245,158,11)" if fired else "rgba(148,163,184,.28)"
    parts.append(f'<path d="{LOOP_PATH}" fill="none" stroke="{lc}" '
                 f'stroke-width="{2 if fired else 1.2}" stroke-dasharray="5 4" '
                 f'marker-end="url(#{"ahl" if fired else "ahd"})"/>')
    parts.append(f'<text x="286" y="250" font-size="8" fill="{lc}" '
                 f'font-weight="{"700" if fired else "400"}">retry</text>')

    # --- nodes ---
    for node in NODE_XY:
        node_runs = runs.get(node, [])
        detail = node_runs[-1].get("detail", "") if node_runs else ""
        parts.append(_svg_node(node, state_of(node), detail, len(node_runs)))

    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# QUERY FLOW  (the expandable per-step detail, shown under the diagram)
# ---------------------------------------------------------------------------
# The STATIC shape of the graph, in reading order.
#   (node, depth, edge label that leads to it)
# Mirrors build_graph() in step9_memory.py.
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
    """The expandable per-step detail that sits UNDER the diagram.

    The SVG above already carries the structure and the highlighted path, so
    this lists only the nodes that actually ran - in execution order (dicts
    keep insertion order) - and exists to be clicked open for outputs.
    """
    runs = runs or {}
    if not runs and not current:
        return ('<div class="fdesc">Ask a question and the path will light '
                'up here.</div>')

    parts = ['<div class="gsteps">']
    for node, node_runs in runs.items():
        state = "failed" if node == "no_answer" else "done"
        last = node_runs[-1]
        seconds = sum(r.get("seconds", 0) or 0 for r in node_runs)

        if len(node_runs) > 1:
            output = "".join(
                f'<div class="frun"><div class="frunhead">attempt {i}</div>'
                f'{r.get("output", "")}</div>'
                for i, r in enumerate(node_runs, start=1))
            detail = f'ran {len(node_runs)}x - {last.get("detail", "")}'
        else:
            output = last.get("output", "")
            detail = last.get("detail", "")

        parts.append(_card(node, state, detail, seconds, output))

    if current:
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


# ---------------------------------------------------------------------------
# THE WHOLE DRAWER, AS ONE BLOCK
# ---------------------------------------------------------------------------
def drawer_html(runs: dict | None = None, current: str | None = None,
                route: str | None = None, finished: bool = False,
                ingest: dict | None = None,
                sources: list | None = None, engine: str = "",
                thread_id: str = "") -> str:
    """Everything in the panel: styles, diagram, step outputs, ingestion.

    Returned as ONE html string so a single st.markdown owns the entire
    subtree. Nothing inside is a Streamlit element, so there are no nested
    block wrappers to fight with when the panel is position:fixed.
    """
    ingest = ingest or {}
    seconds = ingest.get("seconds", 0.0)

    return (
        styles()
        + '<div class="ragdrawer">'
        + '<div class="dtitle">Pipeline</div>'
        + '<div class="dsub">What the graph did for the latest question.</div>'
        + '<div class="dsec">The graph</div>'
        + graph_svg(runs, current, route, finished)
        + '<div class="dsec">Step outputs &mdash; click to expand</div>'
        + query_html(runs, current, route, finished)
        + '<div class="dsec">Startup &mdash; ingestion, ran once</div>'
        + ingestion_html(ingest.get("current"), ingest.get("done"),
                         ingest.get("facts"))
        + f'<div class="fnote">Ran once in {seconds:.1f}s, not per question. '
          'Embedding is ~90% of it.</div>'
        + '<div class="dsec">Documents</div>'
        + "".join(f'<div class="dfile">{_esc(f)}</div>'
                  for f in (sources or []))
        + '<div class="dsec">Engine</div>'
        + _kv("model", engine)
        + _kv("search", "fastembed / bge-small-en-v1.5")
        + _kv("thread", thread_id)
        + '</div>'
    )


# ---------------------------------------------------------------------------
# CHAT BUBBLES
# ---------------------------------------------------------------------------
def chat_css() -> str:
    """Assistant on the left, you on the right.

    Streamlit renders every chat message identically left-aligned. We flip the
    user's by selecting on the avatar Streamlit puts inside each message:
    `.stChatMessage:has(.stChatMessageAvatarUser)`. The :has() parent selector
    is what makes this possible without wrapping every message ourselves -
    supported in all current browsers.

    Class names verified against the installed Streamlit bundle rather than
    assumed: stChatMessage, stChatMessageAvatarUser,
    stChatMessageAvatarAssistant, stChatMessageContent.
    """
    return """
<style>
/* --- your messages: right --- */
.stChatMessage:has(.stChatMessageAvatarUser) {
    flex-direction: row-reverse;
    margin-left: auto;
    margin-right: 0;
    max-width: 78%;
    background: rgba(16,185,129,.10);
    border: 1px solid rgba(16,185,129,.22);
    border-radius: 16px 16px 4px 16px;
    padding: .35rem .75rem;
}
.stChatMessage:has(.stChatMessageAvatarUser) .stChatMessageContent {
    text-align: right;
}

/* --- the assistant: left --- */
.stChatMessage:has(.stChatMessageAvatarAssistant) {
    margin-right: auto;
    margin-left: 0;
    max-width: 88%;
    background: rgba(148,163,184,.07);
    border: 1px solid rgba(148,163,184,.16);
    border-radius: 16px 16px 16px 4px;
    padding: .35rem .75rem;
}
</style>
"""
