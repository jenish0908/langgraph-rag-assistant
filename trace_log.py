"""
Per-question trace logging.

WHY THIS EXISTS
---------------
The nodes used to `print()`. That is fine for watching a run and useless for
reviewing one: it scrolls away, it isn't timestamped, and it drops the details
you actually need when an answer looks wrong - the full chunk text, the
grader's raw output, the similarity scores.

Two outputs, one system:

  logs/assistant.log    human-readable, full detail, appended forever
  logs/questions.jsonl  one JSON object per question, for analysis across runs

The console still shows a brief version. That is what log LEVELS are for:
  INFO  -> console + file   (the short version)
  DEBUG -> file only        (chunk text, raw model output, token counts)
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

TEXT_LOG = LOG_DIR / "assistant.log"
JSONL_LOG = LOG_DIR / "questions.jsonl"

_WIDTH = 100


# ---------------------------------------------------------------------------
# LOGGER SETUP
# ---------------------------------------------------------------------------
def _build_logger() -> logging.Logger:
    log = logging.getLogger("assistant")
    if log.handlers:              # already configured - don't double up
        return log
    log.setLevel(logging.DEBUG)
    log.propagate = False

    # File: everything, with timestamps.
    fh = logging.FileHandler(TEXT_LOG, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s",
                                      datefmt="%H:%M:%S"))
    log.addHandler(fh)

    # Console: the short version, no timestamps, ASCII-safe for cp1252.
    # stdout, not the default stderr - so `python x.py 2>/dev/null`
    # (used to hide library warnings) doesn't hide the trace too.
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(ch)
    return log


logger = _build_logger()


def _cite(doc) -> str:
    """Short label for a chunk, including the PDF page when there is one."""
    m = doc.metadata
    page = f" p.{m['page']}" if "page" in m else ""
    return f"{m['source']}{page}#{m['chunk_id']}"


def _safe(text: str) -> str:
    """Strip characters the Windows console cannot encode (cp1252)."""
    return text.encode("ascii", "replace").decode("ascii")


# ---------------------------------------------------------------------------
# THE TRACE - one per question
# ---------------------------------------------------------------------------
class QuestionTrace:
    """Collects everything that happens while answering ONE question."""

    _counter = 0

    def __init__(self, user_message: str, thread_id: str):
        QuestionTrace._counter += 1
        self.n = QuestionTrace._counter
        self.user_message = user_message
        self.thread_id = thread_id
        self.started = time.time()
        self.started_iso = datetime.now().isoformat(timespec="seconds")
        self.events: list[dict] = []      # structured record for the JSONL
        self.answer = ""
        self.route = ""
        self.attempts = 0

        logger.info("")
        logger.info("=" * _WIDTH)
        logger.info(f"QUESTION #{self.n}  |  thread={thread_id}  |  {self.started_iso}")
        logger.info("=" * _WIDTH)
        logger.info(f"USER MESSAGE : {_safe(user_message)}")

    # -- helpers ------------------------------------------------------------
    def _elapsed(self) -> float:
        return time.time() - self.started

    def _head(self, node: str, seconds: float, extra: str = "") -> None:
        tail = f"  {extra}" if extra else ""
        logger.info(f"\n[{node}]  {seconds:.1f}s  (t+{self._elapsed():.1f}s){tail}")

    # -- one method per node ------------------------------------------------
    def contextualize(self, seconds, history_len, standalone, rewritten):
        self._head("contextualize", seconds)
        if rewritten:
            note = ""
        elif history_len == 0:
            note = "   (first turn - no history to resolve)"
        else:
            note = "   (already self-contained)"
        logger.info(f"  standalone     : {_safe(standalone)!r}{note}")
        logger.debug(f"  history seen   : {history_len} messages")
        self.events.append({"node": "contextualize", "seconds": round(seconds, 2),
                            "history_messages": history_len,
                            "standalone_question": standalone,
                            "rewritten": rewritten})

    def classify(self, seconds, raw, route):
        self._head("classify", seconds)
        logger.info(f"  route          : {route}")
        logger.debug(f"  raw output     : {raw.strip()!r}")
        self.route = route
        self.events.append({"node": "classify", "seconds": round(seconds, 2),
                            "raw": raw.strip(), "route": route})

    def retrieve(self, seconds, attempt, query, scored):
        """scored: list of (Document, score) straight from the vector store."""
        self._head("retrieve", seconds, f"attempt {attempt}")
        logger.info(f"  query          : {_safe(query)!r}")
        logger.info(f"  results        : " + ", ".join(
            f"{_cite(d)}({s:.3f})" for d, s in scored))
        # Full chunk text goes to the FILE only - this is the bit you need
        # when an answer looks wrong.
        for i, (d, score) in enumerate(scored, start=1):
            logger.debug(f"    [{i}] {_cite(d)}"
                         f"  score={score:.4f}  chars={len(d.page_content)}")
            for line in _safe(d.page_content).strip().splitlines():
                logger.debug(f"        | {line}")
        self.attempts = attempt
        self.events.append({
            "node": "retrieve", "seconds": round(seconds, 2), "attempt": attempt,
            "query": query,
            "results": [{"source": d.metadata["source"],
                         "chunk_id": d.metadata["chunk_id"],
                         "score": round(float(s), 4),
                         "chars": len(d.page_content)} for d, s in scored],
        })

    def grade(self, seconds, attempt, raw, docs, verdicts):
        kept = sum(verdicts)
        self._head("grade_docs", seconds, f"attempt {attempt}")
        logger.info(f"  raw output     : {raw.strip()!r}")
        logger.info(f"  kept           : {kept}/{len(docs)}")
        for i, (d, ok) in enumerate(zip(docs, verdicts), start=1):
            mark = "KEEP" if ok else "drop"
            logger.info(f"    [{i}] {_cite(d)}  -> {mark}")
        self.events.append({
            "node": "grade_docs", "seconds": round(seconds, 2), "attempt": attempt,
            "raw": raw.strip(), "kept": kept, "total": len(docs),
            "verdicts": [{"source": d.metadata["source"],
                          "chunk_id": d.metadata["chunk_id"], "keep": bool(ok)}
                         for d, ok in zip(docs, verdicts)],
        })

    def rewrite(self, seconds, old_query, new_query, tried):
        self._head("rewrite_query", seconds)
        logger.info(f"  new query      : {_safe(new_query)!r}")
        logger.debug(f"  previous       : {_safe(old_query)!r}")
        logger.debug(f"  already tried  : {tried}")
        self.events.append({"node": "rewrite_query", "seconds": round(seconds, 2),
                            "previous_query": old_query, "new_query": new_query,
                            "tried": list(tried)})

    def generate(self, seconds, docs, answer, meta=None):
        self._head("generate", seconds)
        logger.info(f"  chunks used    : " + ", ".join(_cite(d) for d in docs))
        if meta:
            logger.info(f"  tokens         : in={meta.get('prompt_eval_count')} "
                        f"out={meta.get('eval_count')}")
            logger.debug(f"  read           : "
                         f"{meta.get('prompt_eval_duration', 0) / 1e9:.1f}s")
            logger.debug(f"  write          : "
                         f"{meta.get('eval_duration', 0) / 1e9:.1f}s")
        self.events.append({
            "node": "generate", "seconds": round(seconds, 2),
            "chunks_used": [f"{d.metadata['source']}#{d.metadata['chunk_id']}"
                            for d in docs],
            "prompt_tokens": (meta or {}).get("prompt_eval_count"),
            "output_tokens": (meta or {}).get("eval_count"),
        })

    def chat_reply(self, seconds, answer):
        self._head("chat_reply", seconds)
        self.events.append({"node": "chat_reply", "seconds": round(seconds, 2)})

    def no_answer(self, attempts):
        self._head("no_answer", 0.0, f"gave up after {attempts} attempts")
        self.events.append({"node": "no_answer", "attempts": attempts})

    # -- close it out -------------------------------------------------------
    def finish(self, answer: str) -> None:
        self.answer = answer
        total = self._elapsed()
        logger.info("-" * _WIDTH)
        logger.info(f"RESULT   route={self.route}  attempts={self.attempts}  "
                    f"total={total:.1f}s")
        logger.info(f"ANSWER   {_safe(answer)}")
        logger.info("=" * _WIDTH)

        record = {
            "n": self.n,
            "timestamp": self.started_iso,
            "thread_id": self.thread_id,
            "user_message": self.user_message,
            "route": self.route,
            "attempts": self.attempts,
            "total_seconds": round(total, 2),
            "answer": answer,
            "events": self.events,
        }
        with open(JSONL_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# HOW NODES REACH THE CURRENT TRACE
#
# The trace cannot live in graph state: the SQLite checkpointer would try to
# serialise it. A module-level global would break under Streamlit, which
# serves requests from a thread pool - two users would overwrite each other.
#
# contextvars is the tool for exactly this: a value that is visible to
# everything in the current execution context, and separate per thread.
# ---------------------------------------------------------------------------
import contextvars

_current: contextvars.ContextVar = contextvars.ContextVar("trace", default=None)


class NullTrace:
    """A trace that does nothing.

    Returned when no trace is active, so nodes can always call
    `current().grade(...)` without an `if`. Cheaper than scattering None
    checks through every node.
    """

    def __getattr__(self, _name):
        def noop(*args, **kwargs):
            return None
        return noop


_NULL = NullTrace()


def start_trace(user_message: str, thread_id: str) -> QuestionTrace:
    """Begin logging a question. Call this before graph.invoke/stream."""
    t = QuestionTrace(user_message, thread_id)
    _current.set(t)
    return t


def current():
    """The trace for the question being answered right now."""
    return _current.get() or _NULL
