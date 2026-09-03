# Agentic RAG Assistant — A Beginner's LangGraph Tutorial

> A "Chat With My Documents" assistant, built one small step at a time.
> Every step: **why we're doing it** → **the code** → **run it and see**.

---

## Table of contents

| Step | What we build | Status |
|------|---------------|--------|
| 0 | Concepts: what RAG is, what LangGraph adds | ✅ done |
| 1 | Project setup — venv, dependencies, folders | ✅ done |
| 2 | Hello Graph — 3 fake nodes, zero magic | ✅ done |
| 3 | Load & chunk documents | ✅ done |
| 4 | Embed & store in a vector store | ✅ done |
| 5 | Wire `retrieve` + `generate` — first working RAG | ✅ done |
| 6 | Add `grade_docs` — first conditional edge | ✅ done |
| 7 | Add `rewrite_query` — the self-correcting loop | ✅ done |
| 8 | Add `route` + a Streamlit chat UI | ✅ done |
| 9 | Conversation memory — checkpointer + follow-up questions | ✅ done |
| 10 | Trace logging — a full record of every question | ✅ done |

---

# Step 0 — The concepts

## The problem

An LLM only knows what was in its training data. It knows nothing about *your*
PDFs, notes, or company docs. Ask it about them and it either says
"I don't know" or — worse — invents a confident, wrong answer (a *hallucination*).

## RAG: Retrieval-Augmented Generation

RAG fixes this in three moves:

1. **Chop** — split your documents into small chunks (a few hundred words each).
2. **Index** — convert each chunk into a list of numbers (an **embedding**) that
   captures its *meaning*, and store those numbers.
3. **Retrieve + Generate** — at question time, find the chunks whose meaning is
   closest to the question, paste them into the prompt, and let the LLM answer
   **from that real text** instead of from memory.

### What is an embedding?

A model reads a chunk of text and outputs ~384 numbers — a *vector*. Chunks with
similar meaning end up with similar vectors, so "How do I reset my password?"
lands near "Password recovery steps" even though they share almost no words.

That's the magic: we search by **meaning**, not by keyword.

## Why "agentic" RAG — this is the LangGraph part

Plain RAG is a straight line: always search → always answer. That's dumb in
three specific ways, and we fix each one with a **node** in a graph:

| Dumb behavior | Our fix |
|---|---|
| Searches your PDFs even for "hi, how are you" | a `route` node that decides *whether* to search |
| Answers confidently from irrelevant chunks | a `grade_docs` node that checks the chunks are actually on-topic |
| Gives up when your question was worded badly | a `rewrite_query` node that rephrases and **loops back** to try again |

That **loop back** is the thing a normal chain physically cannot do.
It is the entire reason LangGraph exists.

## The graph we're heading toward

```
        question
            |
         [route] ------ no search needed ---> [answer] ---> END
            |
         search
            v
       [retrieve]  <----------------+
            |                       |
      [grade_docs]                  |
            |                       |
      +-----+------+          [rewrite_query]
   relevant    irrelevant           |
      |        (tries < 3) ---------+
      v
  [generate] ---> END
```

We will not build this all at once. We build the straight line first
(Steps 3-5), then bolt on the smart parts (Steps 6-8).

---

# Step 1 — Project setup

## Why we do this first

### The virtual environment

A **virtual environment** (`venv`) is a private box of Python packages that
belongs to this project alone. Without one, every `pip install` goes into your
system Python — and eventually two projects need conflicting versions of the
same package and both break.

It costs one command. It is the single best habit to build early.

```
rag_assistant/
  .venv/          <- the private box. Never edit, never commit to git.
  docs/           <- you drop your PDFs / .md files here
  requirements.txt
```

### Why these dependencies

| Package | What it does | Why we chose it |
|---|---|---|
| `langgraph` | Builds the state machine — nodes, edges, loops | The whole point of the project |
| `langchain-core` | Shared types: `Document`, messages, vector-store interface | Everything else speaks these types |
| `langchain-anthropic` | Talks to Claude | Our LLM for grading + answering |
| `langchain-text-splitters` | Chops documents into chunks | Step 3 needs it |
| `fastembed` | Turns text into embeddings, **locally** | Free, no API key, ~50 MB — vs. PyTorch's ~2 GB |
| `pypdf` | Pulls text out of PDFs | Reading your source files |
| `python-dotenv` | Loads your API key from a `.env` file | Keeps secrets out of your code |

**The embedding choice, explained.** Embeddings need a model. You can either
call a cloud API (costs money, needs a key) or run a small model on your own
machine. We run it locally with `fastembed` — free, private, no key, and it
uses ONNX instead of PyTorch so the download is ~50 MB rather than ~2 GB.

Only the **answering** step uses Claude, so that's the only place you need
an API key. Steps 1-4 need no key at all.

## The commands

```bash
# 1. Create the project folder and a place for your documents
mkdir rag_assistant
cd rag_assistant
mkdir docs

# 2. Create the virtual environment
python -m venv .venv

# 3. Activate it
#    Windows (PowerShell):
.venv\Scripts\activate
#    Windows (Git Bash):
source .venv/Scripts/activate
#    macOS / Linux:
source .venv/bin/activate

# You'll know it worked: your prompt now starts with (.venv)

# 4. Install everything
pip install -r requirements.txt
```

## `requirements.txt`

```
langgraph
langchain-core
langchain-anthropic
langchain-community
langchain-text-splitters
fastembed
pypdf
python-dotenv
```

## Checkpoint

Run this — it should print the LangGraph version with no errors:

```bash
python -c "import langgraph; print('LangGraph OK')"
```


## What actually got installed

```
langgraph                    1.2.11
langchain-core               1.6.1
langchain-anthropic          1.7.0
langchain-text-splitters     1.1.2
fastembed                    0.8.0
pypdf                        6.16.2
python-dotenv                1.2.3
```

### A note on `langchain-community`

The first draft of `requirements.txt` included `langchain-community`, because
that's where LangChain's FastEmbed wrapper lives. On import it printed:

```
DeprecationWarning: `langchain-community` is being sunset and is no longer
actively maintained.
```

So we **dropped it**. In Step 4 we'll write our own ~10-line adapter that plugs
`fastembed` into LangChain directly. That's better for you anyway — you'll see
exactly what an "embeddings object" is instead of treating it as a black box.

**Lesson worth keeping:** in a fast-moving ecosystem, a deprecation warning is
information, not noise. Read them.

## Supporting files we added

### `.gitignore`
Keeps junk and secrets out of version control:
```
.venv/          <- 100s of MB of packages; recreate with requirements.txt instead
.env            <- YOUR API KEY. Never, ever commit this.
__pycache__/
docs/*          <- your personal PDFs shouldn't go to GitHub
!docs/.gitkeep  <- ...but keep the empty folder itself
```

### `.env.example`
A template that IS committed, showing what keys are needed without exposing
real ones. Anyone cloning the repo copies it to `.env` and fills in their own.
```
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxx
```

## ✅ Step 1 checkpoint

Final structure:

```
rag_assistant/
├── .venv/            (the private package box)
├── docs/             (drop your PDFs / .md files here)
├── .env.example      (template for secrets)
├── .gitignore
├── requirements.txt
└── TUTORIAL.md       (these notes)
```

Verify it works:

```bash
python -c "import langgraph; print('LangGraph OK')"
```

No API key needed yet. Steps 2, 3 and 4 also need no key —
the first Claude call comes in Step 5.

---

# Step 2 — Hello Graph (the four core concepts)

**File:** `step2_hello_graph.py` · **Run:** `python step2_hello_graph.py`
**Needs an API key:** no.

## Why a *graph* at all?

You already know how to do this:

```python
text    = clean(text)
count   = count_words(text)
summary = summarize(text)
```

That's a **chain** — a straight line, fixed the moment you type it.
It works fine until you need:

- **Branching** — "if the docs are irrelevant, do something else"
- **Cycles** — "try again, up to 3 times"
- **Pausing** — stop mid-run, ask a human, resume later
- **Inspection** — see exactly what changed after each step
- **Resuming** — it crashed at step 4; restart from step 4, not step 1

You *can* hand-code all of that with `if`, `while`, and a pile of variables.
It becomes unreadable fast, and you can't visualize or persist it.

LangGraph asks you to describe your program as a **graph**: boxes (nodes)
connected by arrows (edges), with one shared bag of data flowing through.
Once your program is *data* instead of *control flow*, the framework can do
the loops, pauses, saves and diagrams for you.

> **One-line version:** a chain is a recipe you follow top to bottom.
> A graph is a map you navigate — you decide at each junction which way to
> turn, and you're allowed to drive in circles.

---

## Concept 1 — State: the shared bag

There is exactly **one dictionary** flowing through your entire graph.
Every node reads it; every node may write to it.

```python
from typing import TypedDict

class State(TypedDict):
    text: str
    word_count: int
```

**Why `TypedDict` and not a normal class?** At runtime it *is* a plain `dict`
— no overhead, no special object. The annotations exist so (a) your editor
autocompletes `state["text"]` and catches typos, and (b) LangGraph reads them
to learn the field names and how to merge updates. It's a schema, not a
container.

### The rule beginners always get wrong

A node receives the **whole** state but returns **only the keys it changed**.

```python
def count_words(state: State):
    n = len(state["text"].split())
    return {"word_count": n}                     # correct
    # return {"text": ..., "word_count": n}      # wrong - don't echo everything
```

LangGraph merges that small dict into the main state. Keys you don't mention
are left alone. Mentally: `state.update(whatever_the_node_returned)`.

This is why nodes stay small and independent — each minds its own field.

---

## Concept 2 — Reducers: *how* updates get merged

Merging sounds trivial, but there's a real choice hiding in it.

State has `log: list[str]`, and a node returns `{"log": ["cleaned"]}`. Should that:

- **Replace** the old list -> `["cleaned"]`, or
- **Append** to it -> `["loaded", "cleaned"]`?

**By default LangGraph replaces.** Last write wins. For `text` and
`word_count` that's exactly right.

But for a running log — or retrieved documents, or chat history — you want to
*accumulate*. Ask for that with `Annotated`:

```python
import operator
from typing import Annotated, TypedDict

class State(TypedDict):
    text: str                                # replace (default)
    word_count: int                          # replace (default)
    log: Annotated[list[str], operator.add]  # APPEND  <- a reducer
```

The second slot in `Annotated[...]` is the **reducer**: a function
`(old, new) -> merged`. `operator.add` on two lists is concatenation, so
returning `{"log": ["cleaned"]}` appends instead of overwriting.

**Why this matters more than it looks:** in Step 7 our graph *loops*. Without
a reducer, each retry would wipe out what the previous attempt learned.
Reducers are how a graph accumulates knowledge across a cycle. You'll also
meet `add_messages`, a purpose-built reducer for chat history.

---

## Concept 3 — Nodes: just functions

No base class to inherit, no required decorator.

```python
def clean_text(state: State) -> dict:
    return {"text": state["text"].strip().lower()}
```

**The contract, in full:**
- **Takes** one argument: the current state dict.
- **Returns** a dict of updates — or `None` / `{}` to change nothing.

A node can call an LLM, hit a database, or just print. LangGraph doesn't care
what's inside.

> **Important:** not every node needs an LLM. Beginners reflexively make
> everything an LLM call. A plain-Python node is faster, free and
> deterministic. Use the LLM only where you genuinely need *judgment*.

---

## Concept 4 — Edges: the arrows

```python
builder.add_edge(START, "clean")     # entry point
builder.add_edge("clean", "count")
builder.add_edge("count", "report")
builder.add_edge("report", END)      # exit
```

`START` and `END` are sentinel values imported from `langgraph.graph`.
`START` isn't a node you write — it's a marker meaning "the graph begins here."
Same idea for `END`.

Nodes are registered **by string name**, and edges refer to those strings.
That indirection is what lets a graph be inspected, drawn and serialized
before it ever runs.

This step uses only plain edges — a straight line. **Conditional** edges (the
branching kind) arrive in Step 6, once the straight line works.

---

## Build -> compile -> run

Three distinct phases; the middle one confuses people.

```python
builder = StateGraph(State)            # 1. BUILD   - declare nodes and edges
graph   = builder.compile()            # 2. COMPILE - validate and freeze
graph.invoke({"text": "  Hello  "})    # 3. RUN
```

**Why is `compile()` separate?** Building is mutable bookkeeping. Compiling is
where LangGraph validates the whole thing at once — every edge points at a
node that exists, there's a reachable entry point, no orphans. Catching a
mistyped node name *there*, before execution, beats crashing halfway through.
Compile returns a frozen runnable object; later it's also where you attach a
checkpointer for memory.

**Two ways to run it:**

| Call | Behaviour |
|---|---|
| `graph.invoke(state)` | Runs to completion, returns the **final** state |
| `graph.stream(state, stream_mode="updates")` | Yields after **each node**, showing what that node returned |

`stream` is your debugger. When a graph misbehaves, it shows you exactly which
node wrote the wrong value.

---

## What we built

```
START -> [clean] -> [count] -> [report] -> END
```

| Node | Demonstrates |
|---|---|
| `clean` | overwriting an existing key (`text`) |
| `count` | writing a brand-new key (`word_count`) |
| `report` | reading a key another node wrote |
| all three | a reducer accumulating into `log` |

## The output

```
RUN 1 - invoke():  runs to completion, returns final state
  [clean]  running...
  [count]  running...
  [report] running...

Final state:
  text         = 'langgraph makes agents easy'
  word_count   = 4
  summary      = "The text has 4 words: 'langgraph makes agents easy'"
  log          = ['cleaned the text', 'counted 4 words', 'built the report']

RUN 2 - stream():  peek at each node's output as it happens
  node 'clean'  returned -> {'text': 'langgraph makes agents easy', 'log': ['cleaned the text']}
  node 'count'  returned -> {'word_count': 4, 'log': ['counted 4 words']}
  node 'report' returned -> {'summary': "...", 'log': ['built the report']}
```

### Read that output carefully

1. Each node returned **only 2 keys**, never the whole state.
2. `text` was **replaced** by `clean` (default reducer).
3. `log` **accumulated** all three entries (`operator.add` reducer) — the
   difference between the two behaviours, visible in one run.
4. `report` could read `word_count`, which `count` wrote. State is how nodes
   talk to each other; they never call one another directly.

---

## Clarification: what "reading keys other nodes wrote" means

**"Keys"** = the dictionary keys in the state.
**"Wrote"** = an earlier node returned that key, so LangGraph merged it in.

The point: `report` uses a value it **did not compute itself**.

### Watch the state dict grow

The state starts as whatever you pass to `invoke()`:

```python
{"text": "   LangGraph Makes Agents EASY   ", "log": []}
```

Note what's *missing*: no `word_count`, no `summary`. They're declared in the
`State` schema, but nothing has written them yet.

**After `clean` returns** `{"text": "langgraph makes agents easy", "log": ["cleaned the text"]}`:

```python
{
  "text": "langgraph makes agents easy",   # <- clean overwrote this
  "log":  ["cleaned the text"],
}
```

**After `count` returns** `{"word_count": 4, "log": ["counted 4 words"]}`:

```python
{
  "text":       "langgraph makes agents easy",
  "word_count": 4,                            # <- count WROTE this key. It's new.
  "log":        ["cleaned the text", "counted 4 words"],
}
```

**Now `report` runs.** Look at its first line:

```python
def report(state: State) -> dict:
    line = f"The text has {state['word_count']} words: {state['text']!r}"
    #                      ^^^^^^^^^^^^^^^^^^^          ^^^^^^^^^^^^^^
    #                      written by `count`           written by `clean`
```

`report` never counts anything. It reaches into the shared bag and finds
`word_count` already sitting there, because `count` dropped it in two steps
earlier. That is "reading a key another node wrote".

### Why this deserves its own concept

In normal Python you'd pass values directly:

```python
n    = count_words(text)
line = report(text, n)      # you hand `n` over explicitly
```

In LangGraph, nodes **never call each other and never pass arguments to each
other**. `report` doesn't know `count` exists — no import, no function call,
no parameter.

They communicate *only* by leaving values in the shared state. Like colleagues
who never speak, but write on the same whiteboard.

### Two real consequences

1. **Order matters, and the edges decide it.** `report` works only because the
   edges guarantee `count` ran first. That's exactly why experiment #3 crashes:
   wire `clean -> report` directly, skipping `count`, and you get
   `KeyError: 'word_count'` because nobody ever wrote that key.

2. **Nodes stay swappable.** Rewrite `count` to count characters instead of
   words, or replace it entirely with an LLM call — `report` needs zero
   changes. It only cares that *something* put a number under `word_count`.
   This is what makes graphs easy to modify later, and why in Step 6 we can
   drop a new node into the middle of the pipeline without touching its
   neighbours.

---

## Gotcha: Windows console encoding

The first run printed `RUN 1 ? invoke()` instead of `RUN 1 — invoke()`.

The Windows terminal defaults to the **cp1252** codepage, not UTF-8, so an
em-dash in a `print()` string can't be encoded. Fixes, in order of preference:

1. Keep console output plain ASCII (`-` instead of the em-dash) — what we did.
2. `set PYTHONUTF8=1` before running.
3. `chcp 65001` to switch the terminal to UTF-8.

This only affects **printing to the terminal**. Reading and writing files with
`encoding="utf-8"` is unaffected, which is why this file keeps its em-dashes
happily.

---

## Try it yourself

Small experiments that make the concepts stick:

1. **Break the reducer.** Change `log: Annotated[list[str], operator.add]` to
   plain `log: list[str]` and re-run. The final log holds only
   `['built the report']` — the last write won. That one change *is* the
   reducer concept.
2. **Return an unknown key.** Make `count` return `{"banana": 1}`. LangGraph
   complains: keys must exist in the state schema.
3. **Rewire the graph.** Point `clean` straight at `report` and delete the
   `count` edges. It crashes inside `report` with `KeyError: 'word_count'` —
   proof that nodes communicate *only* through state.
4. **Add a node.** Write `shout`, which uppercases `summary`, and insert it
   between `report` and `END`.

## ✅ Step 2 checkpoint

You should be able to answer these without looking:

- Where does a node get its input, and what should it return?
- What is the default merge behaviour, and how do you change it?
- What does `compile()` buy you?
- When would you use `stream()` instead of `invoke()`?

**Next:** Step 3 — load real documents from `docs/` and chop them into chunks.

---

# Step 3 — Load & chunk documents

**File:** `step3_load_and_chunk.py` · **Run:** `python step3_load_and_chunk.py`
**Needs an API key:** no. This is pure text processing.

## The `Document` — LangChain's universal container

Everything downstream speaks one type:

```python
Document(
    page_content = "Employees accrue 1.75 days of paid leave per month...",
    metadata     = {"source": "handbook.md", "chunk_id": 3},
)
```

Two fields:

- **`page_content`** — the text that gets embedded and searched.
- **`metadata`** — a free-form dict that rides along untouched.

**Don't skim past `metadata`.** It's how you get citations. When the assistant
answers, it needs to say *"according to handbook.md"* — and that's only
possible if the source filename travelled with the text from the moment it was
loaded. Metadata is baggage the pipeline carries for free; put anything in it
you'll want at answer time.

---

## Why chunk at all? Three separate reasons

Beginners assume it's only about context-window limits. That's the least
interesting reason.

**Reason 1 — Context limits.** You can't paste a 300-page PDF into a prompt.
Obvious, and increasingly less true as context windows grow.

**Reason 2 — Retrieval precision.** Retrieval returns *whole chunks*. If your
chunk is an entire chapter, the LLM receives 20 pages of which one sentence is
relevant. That buries the answer in noise and wastes tokens. If the chunk is
one paragraph, the LLM gets exactly the useful part.

**Reason 3 — Embedding dilution.** The one people miss, and the most important.

An embedding is **one vector per chunk** — a single point representing the
*average* meaning of all that text. Embed a whole employee handbook and you get
one vector meaning roughly "generic corporate document". It won't sit close to
"how many vacation days do I get?" *or* to "what's the wifi password?" — it's
mush, equidistant from everything.

Embed one paragraph about leave policy and you get a vector that sits *tightly*
next to leave questions.

> **Rule of thumb:** one chunk should be about one idea.
> That is what makes its vector sharp.

---

## The chunking dilemma

| | Too small (100 chars) | Too big (10,000 chars) |
|---|---|---|
| Vector | sharp, precise | mushy, diluted |
| Context | fragments — "It expires after 90 days." *What does?* | plenty |
| Noise | low | high — pages of irrelevance |
| Cost | many chunks to search | wasted prompt tokens |

Neither extreme works. **~1000 characters is the standard starting point** —
roughly a paragraph or two. Big enough to stand alone, small enough to be about
one thing.

**Characters, not tokens.** The splitter counts characters by default. Rough
conversion for English: **1 token ≈ 4 characters**, so `chunk_size=1000` is
about 250 tokens. Good enough to start; swap in a real token counter later if
you're optimising cost.

---

## Overlap — why chunks should repeat themselves

Split strictly at 1000 characters and you'll eventually cut mid-thought:

```
chunk 4: "...employees must submit the request form to their manager"
chunk 5: "at least 14 days in advance. Requests inside 14 days require..."
```

Now ask *"how far in advance do I request leave?"* Chunk 5 has "14 days in
advance" but never says what for. Chunk 4 says what for but not the deadline.
**Neither chunk can answer the question, even though the document plainly
does.**

The fix is `chunk_overlap` — each chunk repeats the last N characters of the
previous one:

```
chunk 4: "...submit the request form to their manager at least 14 days in advance."
chunk 5: "...to their manager at least 14 days in advance. Requests inside 14 days..."
                +---------- repeated from chunk 4 ----------+
```

Now the complete thought exists in at least one chunk. **200 characters (20%)
is the standard default.** You're deliberately trading a little storage and
duplication for not losing ideas at the seams.

---

## Why `RecursiveCharacterTextSplitter`

A naive splitter chops blindly every 1000 characters — straight through the
middle of words and sentences.

The *recursive* one tries a **priority list of separators**, falling back only
when it must:

```python
["\n\n",  # 1. paragraph breaks   <- try hardest to split here
 "\n",    # 2. line breaks
 " ",     # 3. spaces (word boundaries)
 ""]      # 4. give up, cut mid-word
```

It splits on paragraphs first. If a piece is still over `chunk_size`, it
re-splits *that piece* on single newlines. Still too big? Spaces. Only as a last
resort does it cut inside a word.

**Result:** chunks land on natural boundaries, so each one reads as coherent
text instead of starting mid-senten-

---

## The code, in two functions

```python
CHUNK_SIZE    = 1000   # characters
CHUNK_OVERLAP = 200    # 20%

def load_documents(docs_dir) -> list[Document]:
    # .md / .txt  -> one Document per file
    # .pdf        -> one Document PER PAGE, so we can cite a page number
    # metadata={"source": path.name}  <- attached at load time

def chunk_documents(documents) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,   # records where in the original each chunk began
    )
    chunks = splitter.split_documents(documents)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    return chunks
```

Two details worth noticing:

1. **`split_documents()` copies parent metadata onto every child.** Tear one
   Document into 6 chunks and all 6 still carry `source: "handbook.md"`. That's
   why citations survive chunking without any work from you.
2. **PDFs load one Document per page**, not one per file — so `metadata["page"]`
   exists and you can cite "handbook.pdf, page 12".

---

## The output

```
LOADING
  handbook.md           4,568 characters
  product_faq.md        4,213 characters
  TOTAL                 8,781 characters across 2 document(s)

CHUNKING   (size=1000, overlap=200)
  11 chunks created
  smallest: 566 chars
  largest:  975 chars
  average:  857 chars

THE FIRST 5 CHUNKS
  [chunk 0]  handbook.md  (starts at char 0,    818 chars)
  [chunk 1]  handbook.md  (starts at char 628,  827 chars)
  [chunk 2]  handbook.md  (starts at char 1291, 962 chars)
  [chunk 3]  handbook.md  (starts at char 2221, 923 chars)
  [chunk 4]  handbook.md  (starts at char 3001, 739 chars)

PROOF THAT OVERLAP WORKS
  Literally repeated: 190 characters
    -> Employees accrue paid leave at a rate of 1.75 days per completed mon...

METADATA ON EVERY CHUNK
  {'source': 'handbook.md', 'start_index': 0, 'chunk_id': 0}
```

### Read that output carefully

1. **Largest chunk is 975, not 1000.** `chunk_size` is a ceiling, not a target.
   The splitter stops early at a paragraph break rather than padding to exactly
   1000 characters. The 566-char chunk is the tail end of a file.

2. **Chunk 1 starts at character 628, but chunk 0 was 818 characters long.**
   `628 + 818 = 1446`, yet chunk 2 starts at 1291. Those numbers overlap on
   purpose — that gap *is* `chunk_overlap` doing its job. The start indices go
   backwards relative to where the previous chunk ended.

3. **190 characters literally repeated** between chunks 0 and 1 — close to our
   200 setting, slightly less because the splitter snapped to a clean boundary.
   Overlap isn't theoretical; you can see the same sentence in both chunks.

4. **Every chunk kept `source`.** Nobody wrote code to do that. It came free
   from `split_documents()`.

---

## Gotcha: that `?` in the output again

The preview printed `# Northwind Robotics ? Employee Handbook` — the em-dash in
`handbook.md` couldn't be shown in the cp1252 console (same issue as Step 2).

**The file itself is fine.** The text was read correctly with
`encoding="utf-8"`, it's stored correctly, and it will be embedded correctly.
Only the *terminal display* mangles it. Worth internalising: an encoding
problem at print time is not an encoding problem in your data.

---

## Try it yourself

1. **Make chunks tiny.** Set `CHUNK_SIZE = 200`, `CHUNK_OVERLAP = 20`. Chunk
   count jumps from 11 to ~50, and the previews become fragments with no
   context — "It cannot be purchased retroactively." *What can't?* That's
   over-chunking, seen directly.

2. **Make chunks huge.** Set `CHUNK_SIZE = 5000`. You get 2-3 chunks, each
   spanning many unrelated topics. In Step 5 you'll see these retrieve badly
   for everything.

3. **Kill the overlap.** Set `CHUNK_OVERLAP = 0` and re-run. The "PROOF THAT
   OVERLAP WORKS" section reports 0 shared characters, and chunk boundaries now
   fall mid-sentence.

4. **Add your own file.** Drop any `.md`, `.txt` or `.pdf` into `docs/` and
   re-run. No code changes needed. A PDF will also show `page` in its metadata.

## ✅ Step 3 checkpoint

You should be able to answer:

- What are the two fields of a `Document`, and what is each for?
- Give the three reasons we chunk. Which one is about vector quality?
- What problem does `chunk_overlap` solve? Describe the failure without it.
- Why "recursive" splitting instead of cutting every N characters?
- How does a chunk still know which file it came from?

**Next:** Step 4 — turn these 11 chunks into vectors and search them by
meaning. Still no API key; the embedding model runs on your machine.

---

## Addendum: PDF support, tested

Step 3's loader always had a `.pdf` branch, but it was never run against a real
PDF. Untested code paths are usually broken, so we tested it.

### It worked - but the page numbers were going nowhere

PDFs load **one Document per page**, specifically so citations can name a page.
Then `format_docs()` ignored the `page` metadata entirely and emitted
`[it_policy.pdf #6]` - pointing at a whole file, which for a 200-page manual
helps nobody.

The fix is a single shared `cite()` helper, now used by the prompt, the log and
the UI:

```python
def cite(doc: Document) -> str:
    m = doc.metadata
    page = f" p.{m['page']}" if "page" in m else ""
    return f"{m['source']}{page} #{m['chunk_id']}"

# handbook.md #3            (markdown - no pages)
# it_policy.pdf p.1 #11     (PDF - page 1)
```

> **The lesson:** carrying metadata is only half the job. We attached `page` in
> Step 3 and felt organised, but never spent it. Metadata you collect and never
> use is a comment that thinks it's a feature.

### The end-to-end test

We generated a two-page PDF - an IT policy that contains the office wifi
password - and dropped it in `docs/`. That turns the question which failed in
**every previous step** into an answerable one:

```
[retrieve]  it_policy.pdf p.1#6(0.734), it_policy.pdf p.2#7(0.557), handbook.md#4(0.557)
[grade_docs] '1:yes 2:no 3:yes' (2/3 kept)
ANSWER  The primary office wireless network uses the password
        Ferrite-Anchor-88 [it_policy.pdf p.1 #6]. This password is rotated
        every 90 days [it_policy.pdf p.1 #6].
```

And the follow-up resolved correctly too: *"how often does it change?"* ->
*"How often does the primary office wireless network password change?"*

Nothing else changed. No code, no re-index command. **That is the payoff for
the loader and the vector store not knowing anything about file formats.**

### Gotcha: scanned PDFs fail silently

A scanned PDF is a **picture** of text. `pypdf` extracts nothing, and there is
no error - the file loads, produces zero Documents, and the assistant simply
never knows it exists. You would tune chunk sizes for an hour on a document
that was never ingested.

So the loader now counts pages that produced text and says so:

```python
if pages_with_text == 0:
    print(f"  WARNING: {path.name} has {len(reader.pages)} page(s) "
          f"but no extractable text.")
    print(f"           It is probably a SCANNED pdf (an image). "
          f"You need OCR - e.g. ocrmypdf - to use it.")
```

The fix is OCR (`ocrmypdf scanned.pdf searchable.pdf`).

**Quick test before you start:** open the PDF and try to select a sentence with
the mouse. If you can't select it, neither can the loader.

> **The general principle:** when a failure mode produces *nothing* rather than
> an *error*, you have to detect it deliberately. Silence is the hardest bug to
> notice - the same reason `with_structured_output` returning `[]` in Step 6
> was so dangerous.

---

# Step 4 — Embed & store

**File:** `step4_embed_and_store.py` · **Run:** `python step4_embed_and_store.py`
**Needs an API key:** no. The embedding model runs on your machine.
First run downloads ~50 MB into `.model_cache/`; after that it's offline.

This is the conceptual heart of RAG. Take your time here.

---

## What an embedding actually *is*

A model reads text and outputs a fixed-length list of numbers:

```
"paid leave policy"  ->  [-0.010, -0.015, -0.001, -0.049, 0.040, ...]
                          ^ exactly 384 numbers, every time
```

Always 384 — whether the input is one word or 900 characters. That list is a
**vector**: a coordinate in 384-dimensional space.

The model was trained so that **text with similar meaning lands at nearby
coordinates.** That is the entire trick. It learned this from enormous amounts
of text, which is why "paid leave" ends up near "vacation days" despite sharing
no letters.

**Why 384 dimensions?** You can't capture meaning in 2 or 3. Meaning has many
independent axes at once — formality, topic, sentiment, tense, domain,
specificity. Each dimension is a learned axis. Nobody can tell you what
dimension #212 "means"; the model invented them. 384 is this model's budget.
Bigger models use 768 or 1536 — more nuance, more storage, slower search.

---

## Cosine similarity: measuring "nearby"

Given two vectors, how close are they? **Cosine similarity** — the angle
between them, ignoring length:

```
 1.0  = same direction     -> identical meaning
 0.7+ = small angle        -> strongly related
 0.3  = wide angle         -> vaguely related
 0.0  = perpendicular      -> unrelated
-1.0  = opposite direction
```

**Why angle instead of straight-line distance?** Because vector *length* mostly
tracks text length and emphasis, not meaning. A one-line answer and a
three-paragraph version of the same idea point the same direction but have
different magnitudes. Angle throws that away and keeps only the meaning.

In practice you'll rarely see negative scores with modern models. The useful
range is roughly **0.3 to 0.9**, which is why "0.5 sounds like a coin flip" is
the wrong intuition — see the failure case below.

---

## What a vector store does

Two jobs, that's all:

1. **Store** every chunk's vector alongside its text and metadata.
2. **Search** — given a query vector, return the *k* closest chunks.

We use **`InMemoryVectorStore`** — essentially a Python list. It compares your
query against all 11 vectors and sorts them.

**Why not Chroma or Pinecone?** With 11 chunks, brute force takes microseconds
and there is nothing to install, configure, or debug. A real store adds three
things you don't need yet:

| Feature | What it buys you | When you need it |
|---|---|---|
| Persistence | survives restart | when re-embedding gets slow |
| Approximate search | a million vectors without a million comparisons | ~100k+ chunks |
| Metadata filtering | `WHERE source = 'handbook.md'` | multi-tenant / scoped search |

Swapping one in later is a two-line change, because they all implement the same
interface. **Learn the interface now; pick a database when you actually have
the problem.**

---

## The `Embeddings` interface — the adapter we promised in Step 1

Any embeddings object in LangChain is just two methods:

```python
class Embeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
```

That's the whole contract. Implement those two and every vector store in
LangChain accepts your object. This is why we could drop `langchain-community`
in Step 1 without losing anything:

```python
class FastEmbedAdapter(Embeddings):
    def __init__(self, model_name=MODEL_NAME, cache_dir=CACHE_DIR):
        from fastembed import TextEmbedding
        self.model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    def embed_documents(self, texts):
        return [vec.tolist() for vec in self.model.embed(texts)]

    def embed_query(self, text):
        return list(self.model.query_embed(text))[0].tolist()
```

Ten lines. That's a whole "integration".

### Why are queries and documents embedded by different methods?

This looks redundant and isn't. Some embedding models are trained
**asymmetrically**: documents are embedded plainly, but queries get a hidden
instruction prefix such as

> "Represent this sentence for searching relevant passages: "

which nudges a *question* toward the *statements that answer it*, rather than
toward other questions. The interface has two methods so a model that needs
this can do it.

### ...and what actually happens with our model

We checked instead of assuming:

```python
t = 'How many vacation days do I get?'
e.embed_documents([t])[0] == e.embed_query(t)   # -> True
```

**Identical.** For `bge-small-en-v1.5`, fastembed's `query_embed()` does *not*
apply the prefix. So we added it by hand and measured:

```
Q: What do I do if my laptop gets stolen?
  no prefix    #4:0.593 | #5:0.556 | #10:0.479
  WITH prefix  #4:0.596 | #5:0.560 | #10:0.489

Q: How precise is the robot arm?
  no prefix    #6:0.708 | #7:0.672 | #10:0.619
  WITH prefix  #6:0.708 | #7:0.651 | #10:0.599
```

Scores moved by about 0.01. No ranking that mattered changed. So we kept the
simple version.

> **The transferable lesson is not "prefixes don't matter."** It's that a
> mechanism you read about in a docstring may not be firing in your setup, and
> a five-line experiment settles it. Verify, don't assume.

---

## The output, part 1: meaning beats keywords

```
B. SIMILAR MEANING -> SIMILAR VECTOR
  [0] How many vacation days do I get?
  [1] What is the annual paid leave allowance?
  [2] How do I reset my laptop password?

  [0] vs [1]  0.576  #######################     same meaning, no shared words
  [0] vs [2]  0.396  ###############             different topic
  [1] vs [2]  0.377  ###############             different topic
```

Sentences 0 and 1 share essentially no content words — "vacation days" vs
"paid leave allowance" — yet score meaningfully higher than either does against
the unrelated sentence. Keyword search would rank 0 and 1 at **zero**.

## The output, part 2: searching real chunks

```
QUERY: 'How much vacation time do I get?'      (docs say "paid leave")
  1. 0.651  [handbook.md #1]  Employees accrue paid leave at a rate of 1.75...

QUERY: 'Can I work from Spain for a month?'    (docs never say "Spain")
  1. 0.598  [handbook.md #2]  ... [+ 3. Remote Work + 4. Expenses]

QUERY: 'How precise is the robot arm?'         (docs say "repeatability")
  1. 0.708  [product_faq.md #6]  ... [+ How accurate is it?]

QUERY: 'What do I do if my laptop gets stolen?'
  1. 0.593  [handbook.md #4]  ... [+ 5. Equipment and Security]
  2. 0.556  [handbook.md #5]  Lost or stolen equipment must be reported to...
```

Every top hit is correct, and **not one of them shares the question's key
vocabulary with the document.** "vacation" -> "paid leave". "Spain" ->
"a country other than your country of employment". "precise" ->
"repeatability". This is the payoff, and it's why RAG uses embeddings instead
of `Ctrl+F`.

---

## Gotcha: don't misread your own retrieval results

This one cost real debugging time while writing the step, and it's worth
copying.

**First attempt** — preview each result by its first 60 characters:

```
1. 0.593  [handbook.md #4]  Client entertainment is reimbursable up to $75...
```

That looks like a *terrible* match for "my laptop got stolen". It isn't. Chunk
4 spans a section boundary: it opens with the tail of the Expenses section and
then contains the **entire Equipment & Security section** — laptops, encryption,
passwords, MFA. The preview was lying.

**Second attempt** — show only the `##` headings inside the chunk. Also
misleading: chunk 5 opens with the "Lost or stolen equipment must be reported"
paragraph *before* its first heading, so a heading-only display hides the very
text that matched.

**The fix is to show both ends:**

```python
opening = preview(doc.page_content, 46)
headings = [ln.lstrip("# ").strip() for ln in lines if ln.startswith("#")]
return f'{opening}  [+ {" + ".join(headings)}]'
```

> **Lesson:** chunks span section boundaries, so no single excerpt represents
> a chunk. If your retrieval "looks wrong", print the *whole* chunk before you
> start tuning parameters. I nearly changed the chunk size to fix a display bug.

This also hints at a real improvement for later: a markdown-aware splitter
(`MarkdownHeaderTextSplitter`) would align chunks to section boundaries so this
straddling never happens.

---

## The failure case — why Step 6 exists

```
QUERY: 'What is the wifi password in the office?'
       (the answer is NOWHERE in our documents)
  1. 0.557  [handbook.md #4]  ... [+ 5. Equipment and Security]
  2. 0.507  [handbook.md #2]  ... [+ 3. Remote Work + 4. Expenses]
  3. 0.494  [handbook.md #1]  Employees accrue paid leave at a rate of 1.75...
```

**Read those scores next to the good ones.** The correct answer for the
vacation question scored 0.651. Complete garbage for the wifi question scores
0.557. That is not a comfortable gap — you cannot reliably separate "right" from
"nothing here" with a score threshold.

The reason is structural:

> A vector store **always** returns the k nearest chunks. It has no concept of
> "nothing here is relevant." **Nearest is not the same as relevant.**

Ask for 3 and you get 3, even from a store containing nothing but cake recipes.

Now imagine handing those three chunks to an LLM with "answer the question
using this context". You have actively set it up to produce a confident, wrong
answer — the exact hallucination RAG was supposed to prevent.

**This is what the `grade_docs` node in Step 6 exists to catch:** a cheap LLM
call that reads the retrieved chunks and answers one question — *do these
actually contain the answer?* Keep this output in mind; we'll rerun this exact
query then.

---

## The `k` parameter

`similarity_search(query, k=3)` — how many chunks come back.

- **k too low (1)** — one near-miss and you have nothing to answer from.
- **k too high (20)** — the real answer is buried among 19 distractors, and
  you pay for every token. LLMs also attend less reliably to the middle of a
  long context.

**k=3 to k=5 is the standard starting range.** We use 3.

---

## Try it yourself

1. **Feel the dimensions.** Embed a single word and a whole paragraph, print
   `len(vec)` for both. Identical. Fixed-size output regardless of input — that
   is what makes vectors comparable at all.

2. **Find the breaking point.** Add queries to the list that get progressively
   further from the documents: "how do I request time off" -> "what's the
   coffee budget" -> "who won the 1998 World Cup". Watch the top score sag from
   ~0.65 to ~0.45. Then try to pick a threshold that cleanly separates good
   from bad. You can't. That's the Step 6 argument in one experiment.

3. **Break the meaning.** Search for a made-up word: `"blorptastic policy"`.
   You still get 3 results with non-trivial scores.

4. **Change `k`.** Set `k=8` in the searches and look at how quickly results
   degrade into noise past the top few.

5. **Prove chunk size matters.** Set `CHUNK_SIZE = 5000` in
   `step3_load_and_chunk.py` and re-run this file. Two or three giant chunks,
   each covering many topics, and the score gap between right and wrong answers
   shrinks. That's **embedding dilution** from Step 3, now measurable.

## ✅ Step 4 checkpoint

You should be able to answer:

- What is an embedding, and why is its length fixed?
- Why cosine *angle* rather than distance?
- What are the only two methods an embeddings object needs?
- Why might queries and documents be embedded differently — and how would you
  check whether that's happening in your setup?
- Why can't you just use a similarity-score threshold to detect
  "this isn't in my docs"?

**Next:** Step 5 — the first Claude call. Feed retrieved chunks into a prompt
and get a real answer with citations. **You'll need an API key from here on:**
copy `.env.example` to `.env` and paste a key from
<https://console.anthropic.com/settings/keys>.

---

# Step 5 — Wire `retrieve` + `generate` (first working RAG)

**Files:** `llm.py`, `step5_first_rag.py` · **Run:** `python step5_first_rag.py`
**Needs an API key:** no — everything runs locally.

---

## Decision: local model instead of Claude

We benchmarked the models already installed via Ollama on this machine
(i7-10510U, 4 cores, **no usable GPU** — Intel UHD doesn't accelerate LLM
inference, so it's all CPU):

| Model | Answer task | Grade "relevant?" | Grade "irrelevant?" | Verdict |
|---|---|---|---|---|
| `gemma3:1b` | 3.1s | ❌ said **no** — wrong | ✅ no | unusable |
| `gemma3:4b` | 8.1s | ✅ yes | ✅ no | **this one** |
| `gemma4` (9.6 GB) | — | — | — | exceeds free RAM, swaps |

`gemma3:1b` failed the *relevance-grading* task — it called a clearly relevant
document irrelevant. That is precisely the node Step 6 is built around, so 1b
is out. Fine for smoke-testing that wiring works, useless for the logic.

### What this buys you

Everything now runs **fully offline** — embeddings *and* the LLM. No API key,
no cost, no data leaving the machine. For a documents assistant that's a
genuinely good property.

### What it costs

~4 tokens/sec on CPU, so ~20-30s per question instead of ~3s. Slow enough to
notice, fast enough to learn with.

---

## `llm.py` — configuration in exactly one place

```python
from langchain_ollama import ChatOllama

MODEL = "gemma3:4b"

def get_llm(max_tokens=300, temperature=0.0):
    return ChatOllama(
        model=MODEL,
        temperature=temperature,
        num_predict=max_tokens,
        keep_alive="10m",
    )
```

Three settings worth understanding:

| Setting | Why |
|---|---|
| `temperature=0` | Randomness. Creative writing wants some; "what does this document say" wants the *most probable* answer, and the same answer twice. Always 0 for factual extraction. |
| `num_predict` | Hard cap on output length. At 4 tok/s, a model that decides to write six paragraphs costs you two minutes. Cap it. |
| `keep_alive="10m"` | How long Ollama holds the model in RAM. Loading gemma3:4b costs ~20s; staying warm makes every later call much faster. |

**Switching to a cloud model later is this entire change:**

```python
from langchain_anthropic import ChatAnthropic

def get_llm(max_tokens=300, temperature=0.0):
    return ChatAnthropic(model="claude-sonnet-5",
                         temperature=temperature, max_tokens=max_tokens)
```

Nothing in the graph changes. Not the nodes, not the edges, not the prompts.
Same lesson as the embeddings adapter in Step 4: **program against the
interface, not the vendor.**

---

## The graph

```
START -> [retrieve] -> [generate] -> END
```

Two nodes, straight line, no branching. Deliberately. Get the straight line
correct, *then* add intelligence — the same reason we built fake nodes in
Step 2 before real ones.

```python
class State(TypedDict):
    question:  str              # what the user asked
    documents: list[Document]   # what retrieve found  <- retrieve writes
    answer:    str              # what generate produced
```

Recognise the shape from Step 2? `generate` reads `documents`, a key it did
not write. Same whiteboard pattern — real work this time.

---

## The one thing that must live OUTSIDE the graph

```python
STORE = build_vector_store()      # module level - runs ONCE at import

def retrieve(state):
    docs = STORE.similarity_search(state["question"], k=K)
```

Put `build_vector_store()` *inside* `retrieve` and you would re-read every
file, re-chunk it, and re-embed all 11 chunks **on every single question**.

It would still work. That is exactly what makes this bug easy to ship — no
error, no wrong answer, just quietly getting slower as your document set grows.

> **Principle:** nodes do *per-request* work. Expensive setup — models,
> connections, indexes — is built once and referenced.

---

## The prompt: three parts

This is where RAG succeeds or fails, and it's just careful instruction-writing.

```
1. INSTRUCTION   "Answer using ONLY the context. If it's not there, say so."
2. CONTEXT       the retrieved chunks, each labelled with its source
3. QUESTION      what the user asked
```

**Why "ONLY the context" matters.** Without it the model blends its training
knowledge with your documents. You'd get plausible answers about *generic*
leave policy rather than *your* leave policy — and you could not tell the
difference by looking. Grounding is the whole point of RAG.

**Why we label each chunk.** `format_docs()` produces:

```
[handbook.md #1]
Employees accrue paid leave at a rate of 1.75 days...

---

[handbook.md #2]
Sick leave is separate from paid leave...
```

The model can only cite what it can see. That `source` metadata we attached
back in Step 3 finally earns its place: **disk -> chunking -> vector store ->
prompt -> citation.** Five steps of carrying it, one payoff.

---

## The output

```
Q: How many paid leave days do I get per year?
  [retrieve] found 3 chunks: handbook.md#1, handbook.md#2, handbook.md#0
  [generate] asking the model... done in 21.3s
A: Employees accrue paid leave at a rate of 1.75 days per completed month of
   service, which equals 21 days per calendar year [handbook.md #1]. ...

Q: How accurate is the AtlasArm A5?
  [retrieve] found 3 chunks: product_faq.md#6, product_faq.md#7, product_faq.md#8
A: Repeatability is +/- 0.03 mm for the A5, measured per ISO 9283 at full
   extension under rated load [product_faq.md #6]. ...

Q: What is the hotel spending limit in London?
A: ... $350 per night in San Francisco, New York, London, Zurich, and
   Tokyo [handbook.md #3]. ...

Q: What is the wifi password in the office?          <- not in the documents
A: I don't know based on the provided documents.      (3.9s)
```

Correct on all four, with citations, entirely offline. **That is a working RAG
system.**

Two caveats on those timings, both explained in the correction below: these
questions had been asked before, so they measure the *cached* path — a genuinely
new question costs 40-60s. And the refusal is fast (3.9s) only because it emits
12 tokens; its prompt still had to be read.

---

## Gotcha: the performance investigation (worth copying as a method)

The **first** run took 66-77s per question. The estimate was 20-30s. Rather
than shrug, we measured — Ollama returns a full timing breakdown in
`response.response_metadata`:

```python
r = llm.invoke(prompt)
m = r.response_metadata
m["prompt_eval_count"]     # tokens IN
m["eval_count"]            # tokens OUT
m["prompt_eval_duration"]  # nanoseconds spent READING the prompt
m["eval_duration"]         # nanoseconds spent WRITING the answer
m["load_duration"]         # nanoseconds spent loading the model
```

Result:

```
prompt tokens     : 749
output tokens     :  81
prompt processing :  0.8s   <- reading 749 tokens of context is nearly free
generation        : 20.1s   <- writing 81 tokens is the whole cost
generation speed  :  4.0 tok/s
```

**Finding 1 (WRONG — corrected in Step 6, keep reading):** reading 749 tokens
took 0.8s while writing 81 took 20s, so output length looked like the entire
cost and context length looked free.

That conclusion was drawn from a **cached** measurement and is false. See
"Correction: what those timings actually measured" at the end of this step.

**Finding 2: the slow run had two models in RAM.** `ollama ps` showed
`gemma3:1b` still resident from the earlier benchmark, sharing 4 cores with
`gemma3:4b`. With only one model loaded, the same questions ran at **20-28s**
— matching the estimate.

```powershell
ollama ps                 # what is loaded right now
ollama stop gemma3:1b     # free it
```

**Finding 3: no thermal degradation.** Five consecutive calls held 3.8-4.5
tok/s. Worth checking on a 15W laptop chip; it wasn't the problem here.

> **The transferable method:** when something is 3x slower than expected, get
> the breakdown before changing anything. The instinct was "shrink the
> context" — measurement showed context was 4% of the cost, and the real
> culprit was an unrelated model squatting in RAM.

---

---

## Correction: what those timings actually measured

While building Step 6 we re-measured with genuinely **new** questions and found
the Step 5 numbers above were misleading. They are left in place, marked, so
the mistake is visible — it is a common one.

### What went wrong

Ollama caches the KV state for prompt prefixes. Ask the *same* question twice
and the second prompt-read is nearly free. Every timing reported above came
from questions that had already been asked in an earlier run, so all of them
measured the **cached** path.

Alternating a cached and an uncached prompt shape makes it obvious:

```
round   cap  in_tok  read_s
    1   600     482    30.1      <- first time this prompt shape is seen
    1   900     658    44.1      <- first time
    1   600     482     0.7      <- now cached
    1   900     658     0.8      <- now cached
    2   600     482     0.7
    2   900     658     0.7
```

**60x difference**, same prompt, same model.

### The honest numbers

Asking questions never asked before:

```
NEW    in=724  read=44.6s  write=15.5s  TOTAL=60.0s
NEW    in=726  read=25.4s  write=16.2s  TOTAL=41.8s
REPEAT in=724  read= 0.7s  write=16.9s  TOTAL=17.9s
```

So the real cost of a **new** question was 40-60s, not the ~20s reported. The
~20s figure was real, but it only applies to a question you have already asked.

### Corrected Finding 1

On this CPU, **prompt processing is roughly 16 tok/s and generation roughly
4 tok/s.** Generation is slower per token, but a RAG prompt has ~700 input
tokens against ~80 output tokens, so:

| | tokens | rate | time |
|---|---|---|---|
| reading the context | ~700 | ~16 tok/s | **~44s** |
| writing the answer | ~80 | ~4 tok/s | ~20s |

**Context length dominates on a cold prompt.** The original advice — "shrink
the answer, not the context" — was exactly backwards. Both matter; context
matters more the first time.

This is why Step 6 truncates documents before grading them: a shorter grading
prompt is the single biggest lever on this hardware.

### A second finding: use all your threads

Ollama defaults to *physical* cores (4 here). Forcing all 8 logical threads,
A/B'd over 6 novel questions:

```
num_thread=4:  prefill 15.8 tok/s   generate 4.2 tok/s   avg total 61.3s
num_thread=8:  prefill 26.6 tok/s   generate 5.9 tok/s   avg total 40.0s
```

A 35% cut for one parameter. It is now set in `llm.py`. Re-measure on other
hardware — more threads is not automatically better.

### The lesson, which is bigger than these numbers

A benchmark that reuses inputs measures your cache, not your system. When you
time anything that might cache — an LLM, a database, an HTTP API, a build —
**vary the input**, or you will confidently publish the wrong number.

I did exactly that, and only caught it because a later measurement disagreed
with an earlier one. When two of your own measurements disagree, one of them is
lying about what it measured.


## The escape hatch, and why it is not enough

We instruct the model: *"If the context does not contain the answer, reply:
I don't know based on the provided documents."*

It worked here. **Do not trust it**, for a structural reason:

Retrieval hands the model three chunks that are *topically adjacent* to the
question. Ask about the wifi password and it receives the security section —
passwords, MFA, hardware keys. The model sees a page about passwords and a
question about a password.

You are asking it to notice an *absence* while staring at something that looks
relevant. Models are bad at that. It will work sometimes — which is exactly
what makes it dangerous. A safeguard that fails 20% of the time silently is
worse than no safeguard, because you stop checking.

**Step 6 replaces that hope with an explicit check.**

---

## Try it yourself

1. **Delete the "ONLY the context" line** from `PROMPT` and ask
   "How many paid leave days do I get?" The model starts blending in generic
   knowledge about leave policy. That one sentence is doing enormous work.

2. **Delete the "I don't know" instruction** and re-ask the wifi question.
   Watch it invent something from the password chunk. This is the
   hallucination RAG exists to prevent, and you can trigger it on demand.

3. **Ask something answerable only by combining both files** — e.g.
   "What laptop rules apply, and what warranty does the arm have?" With `k=3`
   you may not retrieve from both documents. Raise `K` to 6 and try again.
   That is retrieval breadth, felt directly.

4. **Prove output length is the cost.** Change "3 sentences or fewer" to
   "1 sentence" and time it. Then try "a detailed paragraph". Compare.

5. **Break the build-once rule on purpose.** Move `build_vector_store()`
   inside `retrieve()` and time four questions. Now imagine 500 documents.

## ✅ Step 5 checkpoint

You should be able to answer:

- Why is the vector store built at module level rather than inside a node?
- What are the three parts of a RAG prompt, and what does each do?
- How does a citation survive from a file on disk to the final answer?
- Why `temperature=0`?
- On CPU, what dominates response time — reading the context or writing the
  answer? How would you check?
- Why isn't "say I don't know if it's not in the context" a real safeguard?

**Next:** Step 6 — the `grade_docs` node and your first **conditional edge**.
The graph stops being a straight line and starts making decisions.

---

# Step 6 — `grade_docs` and your first conditional edge

**Files:** `step6_grading.py`, `prompts.py` · **Run:** `python step6_grading.py`

The graph stops being a straight line and starts making decisions.

---

## The problem, stated precisely

Step 5 ended with a safeguard built on hope: *"say I don't know if it's not in
the context."* It worked, but the failure mode is structural.

**The real issue is that we merged two jobs into one node.** `generate` was
doing both:

1. judging whether it *can* answer, and
2. answering.

Those pull in opposite directions. The prompt says "be helpful, answer the
question, use this context" — and then, in one clause, "except sometimes
refuse". Helpfulness wins, because that's what the other 95% of the prompt
asks for.

So we split them, and each prompt gets to pull in exactly one direction:

| Node | One job | Prompt bias |
|---|---|---|
| `grade_docs` | *Can* this be answered from these chunks? | **skeptical** |
| `generate` | Answer it. | **helpful** |

> **Generalisable:** when a node behaves unreliably, check whether you've asked
> it to do two conflicting things. Splitting is often the fix — not a better
> prompt for the combined job.

---

## Conditional edges — the actual mechanic

Until now every edge was fixed: `add_edge("retrieve", "generate")` always goes
there. A **conditional** edge asks a function which way to go:

```python
def route_after_grading(state) -> str:      # <- a ROUTER, not a node
    if state["relevant_docs"]:
        return "generate"
    return "no_answer"

builder.add_conditional_edges(
    "grade_docs",               # after this node runs...
    route_after_grading,        # ...call this to decide where to go
    ["generate", "no_answer"],  # ...and it may only return one of these
)
```

### Three things beginners get wrong

1. **A router is not a node.** It takes state and returns a *string* — the name
   of the next node. It must not modify state and shouldn't do real work.
   Nodes do work; routers only choose.

2. **The decision was already made.** `grade_docs` did the thinking and wrote
   its verdict into state. The router just reads it. Keep the LLM call in the
   node and the `if` in the router — mixing them makes both untestable. You
   can unit-test `route_after_grading` with a plain dict, no model required.

3. **You must declare the possible destinations.** That third argument is what
   lets LangGraph validate and draw the graph before it ever runs.

---

## The graph now

```
START -> [retrieve] -> [grade_docs] --relevant--> [generate] -> END
                             |
                        not relevant
                             v
                        [no_answer] -> END
```

`no_answer` is a **plain Python node** — no LLM. It returns a fixed string.
Instant, free, and it cannot hallucinate.

> Worth noticing: the honest answer needs no intelligence at all.

New state field:

```python
class State(TypedDict):
    question:      str
    documents:     list[Document]   # everything retrieve found
    relevant_docs: list[Document]   # NEW: what survived grading
    answer:        str
```

Grading doesn't just decide yes/no — it **filters**. `generate` then sees only
the chunks that passed, so there's less noise in its context.

---

## Design decision: one grading call, not K

We grade all 3 documents in a **single** call, asking for `1:yes 2:no 3:yes`.
Three separate calls would mean three separate prompt reads, and prompt reading
is the dominant cost on this hardware (see the Step 5 correction).

We also truncate each document to 450 characters for grading
(`GRADE_DOC_CHARS`). Judging *relevance* rarely needs the full chunk, and on a
CPU at ~26 tok/s every character cut is time saved.

---

## Gotcha: `with_structured_output()` failed silently

The obvious way to get machine-readable verdicts is a Pydantic schema:

```python
class Grades(BaseModel):
    verdicts: list[bool]

llm.with_structured_output(Grades).invoke(prompt)
```

We tested it against plain text parsing:

```
Q: How many paid leave days do I get per year?   (expect some YES)
  TEXT format   -> '1:yes 2:no 3:yes'     correct
  STRUCTURED    -> verdicts=[]            WRONG

Q: What is the wifi password in the office?      (expect ALL NO)
  TEXT format   -> '1:no 2:no 3:no'       correct
  STRUCTURED    -> verdicts=[]            wrong, but looks right
```

**`with_structured_output` returned an empty list every time — and never
raised.** No exception, no warning, valid-looking object.

Think about what that would have done: an empty verdict list means "nothing is
relevant", so the router always goes to `no_answer`. The system would have
looked like a *working grader that is simply very strict*. It would have passed
the wifi test perfectly. You'd only notice when a user complained that it never
answers anything.

**That is the most dangerous class of bug** — one whose failure output is
plausible.

The plain-text format was correct on every case, in a fraction of the time.

> **Rule of thumb:** small local models are much better at "reply with one line
> in this format" than at filling a JSON schema. On a large cloud model,
> `with_structured_output()` is usually the right choice. Test on *your* model
> rather than following the general advice.

Our parser also fails safe:

```python
found = dict(re.findall(r"(\d+)\s*:\s*(yes|no)", text.lower()))
return [found.get(str(i), "no") == "yes" for i in range(1, expected + 1)]
```

A document the model didn't mention defaults to **drop**, not keep. When the
grader is confused we'd rather say "I don't know" than answer from a chunk
nobody vouched for.

---

## Gotcha: importing a module runs its top-level code

The first version of Step 6 did this:

```python
from step5_first_rag import PROMPT, format_docs
```

Harmless-looking. But `step5_first_rag.py` has this at module level:

```python
STORE = build_vector_store()
LLM   = get_llm(max_tokens=250)
```

So Step 6 silently built the vector store **twice** and created two model
objects. The giveaway was this, at the top of the output:

```
Building the vector store (once)...
Ready.

Building the vector store (once)...     <- twice!
Ready.
```

Nothing crashed. Every answer was correct. It was just quietly doing double
work — exactly the kind of bug that survives to production.

**The fix** was `prompts.py`: a module holding only cheap, shared definitions,
imported by both steps.

> **The rule:** a module other files import should be *safe to import*. Put
> expensive setup behind `if __name__ == "__main__":`, behind a function, or in
> a separate module that holds only definitions.

---

## The output

```
Q: How many paid leave days do I get per year?
  [retrieve]   handbook.md#1, handbook.md#2, handbook.md#0
  [grade_docs] judging... 17.5s -> '1:yes 2:no 3:yes'
               handbook.md#1:KEEP handbook.md#2:drop handbook.md#0:KEEP
  [generate]   answering... 35.2s
A: You accrue paid leave at 1.75 days per completed month of service, which
   equals 21 days per calendar year [handbook.md #0]. ...
   (2/3 chunks kept, 52.9s total)

Q: What is the warranty on the AtlasArm?
  [grade_docs] judging... 19.0s -> '1:yes 2:no 3:yes'
               product_faq.md#8:KEEP product_faq.md#6:drop product_faq.md#10:KEEP
   (2/3 chunks kept, 51.7s total)

Q: What is the wifi password in the office?        <- not in the documents
  [retrieve]   handbook.md#4, handbook.md#2, handbook.md#1
  [grade_docs] judging... 18.8s -> '1:no 2:no 3:no'
               handbook.md#4:drop handbook.md#2:drop handbook.md#1:drop
  [no_answer]  nothing relevant survived grading
A: I don't know based on the provided documents.
   (0/3 chunks kept, 19.4s total)
```

### Read that carefully

1. **On real questions it dropped 1 of 3 chunks.** Retrieval returned three
   nearest chunks; only two actually contained the answer. `generate` now sees
   less noise. That is the filtering working.

2. **On the wifi question it dropped all three** — and the router sent the
   graph to `no_answer`. **The LLM was never asked to answer.**

   That's the whole point. In Step 5 we *hoped* the model would refuse. Now the
   model cannot hallucinate an answer it was never asked to produce. The
   safeguard moved from a request inside a prompt to a structural property of
   the graph.

3. **The failure path is fast** — 19.4s vs ~52s. It skips generation entirely.
   Being honest is cheaper than answering.

---

## Try it yourself

1. **Watch the router choose.** Add `print()` inside `route_after_grading`
   showing what it returns. Run all three questions and watch the path change.

2. **Break the grader deliberately.** Change the grading prompt from "Be
   strict" to "Be generous — if the document is on a related topic, say yes."
   Re-run the wifi question. It'll likely keep the security chunk, reach
   `generate`, and you're back to Step 5's problem. **One word in one prompt
   controls whether the system hallucinates.**

3. **Force the safe-fail path.** Make `parse_verdicts` return `[]` always. The
   graph always answers "I don't know". This is exactly what
   `with_structured_output` was silently doing — see how plausible it looks?

4. **Test the router without a model.** Call it directly:
   ```python
   route_after_grading({"relevant_docs": []})       # -> 'no_answer'
   route_after_grading({"relevant_docs": [1, 2]})   # -> 'generate'
   ```
   Instant, no LLM. That's the payoff for keeping logic out of nodes.

5. **Find a false negative.** Ask something the docs *do* answer but obliquely
   — "Can I keep my laptop if I leave?" or "Is alcohol reimbursable?" Watch
   whether the strict grader drops a chunk it shouldn't. Over-strict grading is
   the real cost of this design, and Step 7 is what fixes it.

## ✅ Step 6 checkpoint

You should be able to answer:

- What's the difference between a node and a router?
- Why did splitting judging from answering fix the reliability problem?
- Why grade all documents in one call rather than one call each?
- Why is a silently-empty structured output more dangerous than a crash?
- Why should `parse_verdicts` default to "drop" rather than "keep"?
- Why is a module with top-level setup code dangerous to import?

**Next:** Step 7 — the **cycle**. Right now a bad question just fails. Instead,
`rewrite_query` will rephrase it and send the graph **back** to `retrieve`, up
to 3 times. That's the loop a plain chain cannot do, and it's where the
`operator.add` reducer from Step 2 finally earns its keep.

---

# Step 7 — `rewrite_query` and the CYCLE

**File:** `step7_rewrite_loop.py` · **Run:** `python step7_rewrite_loop.py`

The graph gains an edge that points **backwards**.

---

## What a cycle is

Every graph so far moved strictly forward:

```
START -> retrieve -> grade_docs -> generate -> END
```

Each node ran **at most once**. That shape is a **DAG** — directed *acyclic*
graph. Even Step 6's conditional edge didn't break it; it chose between two
*forward* paths.

A cycle points backwards:

```
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
```

`retrieve` may now run once, twice, or three times for a single question.

---

## Why we need one

Step 6 had a real weakness. When grading dropped every chunk we concluded
"the answer isn't in the documents". But there are **two** reasons that can
happen:

| Reason | Right response |
|---|---|
| The answer genuinely isn't there (wifi password) | give up |
| The answer *is* there, but the question was worded badly | **try again, differently** |

Step 6 treated both identically. The second case is fixable — do what a person
would do: rephrase and search again.

---

## The loop guard

A cycle can run forever. Every cycle needs a termination condition, and in
LangGraph it lives in state:

```python
MAX_ATTEMPTS = 3

def route_after_grading(state) -> str:
    if state["relevant_docs"]:
        return "generate"          # got what we need - leave the loop
    if state["attempts"] >= MAX_ATTEMPTS:
        return "no_answer"         # tried enough - leave the loop
    return "rewrite_query"         # go around again
```

Three destinations, one of which loops backwards.

> **A cycle with no exit is an infinite loop.
> A cycle with an exit is a retry policy.
> The counter is the entire difference.**

LangGraph also has a backstop — `recursion_limit`, default 25 — which raises
an error rather than spinning forever if you forget your own guard. Treat it
as a smoke alarm, not a fire-suppression system.

---

## Where the Step 2 reducer finally earns its keep

```python
class State(TypedDict):
    original_question: str   # what the user asked - never changes
    question: str            # the CURRENT query - rewritten each lap
    documents: list[Document]                          # replaced each lap

    relevant_docs: Annotated[list[Document], operator.add]   # ACCUMULATES
    tried_queries: Annotated[list[str], operator.add]        # ACCUMULATES

    attempts: int
    answer: str
```

Remember Step 2, where changing `Annotated[list, operator.add]` to plain
`list` made all but the last log entry vanish? In a straight line that barely
mattered — each node ran once.

**In a cycle, `grade_docs` runs up to three times.** Without the reducer, lap 3
would destroy what laps 1 and 2 found. With it, you keep the union of
everything relevant discovered on *any* lap.

That exercise looked like trivia in Step 2. It's load-bearing here.

Two more state design decisions worth copying:

- **`original_question` never changes.** `question` gets rewritten, but
  `generate` answers the *original*. Otherwise the user asks about a paint
  booth and gets an answer to a question they never asked.
- **`documents` is replaced, `relevant_docs` accumulates.** The latest raw
  results are only interesting for the current lap; the *relevant* ones are
  worth keeping forever.

And because accumulation causes repeats, `dedupe()` drops chunks already seen —
duplicated context wastes the prompt budget that dominates our runtime.

---

## The output

### 1. Happy path — the loop never fires

```
Q: How many paid leave days do I get per year?
  [retrieve]     attempt 1: 'How many paid leave days do I get per year?'
                 -> hand#1, hand#2, hand#0
  [grade_docs]   judging... 23.1s -> '1:yes 2:no 3:yes'  (2/3 kept)
  [generate]     answering from 2 chunk(s)... 35.0s
A: You accrue paid leave at 1.75 days per completed month of service, which
   equals 21 days per calendar year [handbook.md #0]. ...
   attempts: 1     58.3s total
```

The cycle exists but costs nothing when it isn't needed.

### 2. The loop fires

```
Q: Can the robot work in a paint booth?
  [retrieve]     attempt 1: 'Can the robot work in a paint booth?'
                 -> prod#7, hand#0, prod#6
  [grade_docs]   judging... 22.5s -> '1:no 2:no 3:no'  (0/3 kept)
  [rewrite]      rephrasing... 7.2s
                 -> 'robot suitability for automotive painting environments'
  [retrieve]     attempt 2: 'robot suitability for automotive painting environments'
                 -> prod#7, prod#6, prod#9
  [grade_docs]   judging... 24.0s -> '1:yes 2:no 3:no'  (1/3 kept)
  [generate]     answering from 1 chunk(s)... 17.5s
A: I don't know based on the provided documents.
   attempts: 2     72.0s total
```

**Read this one carefully — it is more interesting than a success.**

The loop fired correctly. The rewrite ran. Attempt 2 retrieved different chunks
and the grader accepted one, so the router sent it to `generate`.

Then `generate` said **"I don't know"** anyway.

That is not a bug in `generate`. Look at what the grader actually did on
attempt 2. It retrieved `prod#7`, `prod#6`, `prod#9` and replied `1:yes 2:no 3:no`:

| Chunk | Contains the answer? | Grader said | Verdict |
|---|---|---|---|
| `prod#7` — power requirements, programming | no | **yes** | **false positive** |
| `prod#6` — payload, accuracy | no | no | correct |
| `prod#9` — service life, *"must not be operated in explosive atmospheres"* | **YES** | **no** | **false negative** |

So `generate` received only `prod#7` — a chunk about voltages and Python SDKs —
and correctly refused. **The model was right. The grader was wrong twice:** it
approved a useless chunk *and* discarded the one holding the answer.

### Two situations that look identical from outside

**Case A — the grader was wrong and `generate` catches it.** What happened
here. A second, independent check caught the first one's mistake. You want
this.

**Case B — the grader was right but `generate` refuses anyway.** The chunk
genuinely contains the answer and the model still says "I don't know". A real
failure: usually an over-strict answering prompt, or a model too small to
connect the question to the wording in the text.

Both print the same thing. The only way to tell them apart is to look at what
reached `generate`:

```python
def generate(state):
    docs = dedupe(state["relevant_docs"])
    for d in docs:                                  # add this while debugging
        print(f"  --- {d.metadata['source']}#{d.metadata['chunk_id']} ---")
        print(d.page_content[:200])
```

If the answer isn't in that text -> Case A, fix the grader.
If it is -> Case B, fix the answering prompt or the model.

### The cost of grading, stated honestly

Step 6 framed grading as pure upside: it stops hallucinations. This run shows
the other edge. **Grading can throw away the right answer.** `prod#9` was
retrieved successfully, contained the exact sentence needed, and the grader
deleted it.

Every filter trades recall for precision. Ours prevents confident wrong answers
and, at the same rate, produces "I don't know" about things the documents do
say. Which error you prefer depends on the job:

| Application | Worse error | Grade strictly? |
|---|---|---|
| Policy / compliance / legal | a confident wrong answer | yes |
| Medical or safety information | a confident wrong answer | yes, strictly |
| Brainstorming / exploration | an unhelpful refusal | no, or loosely |
| Internal search over your own notes | refusal (you can judge for yourself) | loosely |

Also worth sitting with: **two of the three verdicts on this call were wrong.**
That is `gemma3:4b` — a 4-billion-parameter model — doing a judgment task.
Relevance grading is exactly where model size shows, and a larger model would
grade noticeably better. Our layered design partly compensates for a weak
grader, but it cannot fully rescue one.

> **Two independent checks, and the second caught the first one's mistake.**
> Layered defences work because the layers fail differently. But layering is
> not free: each layer that can reject also adds a way to lose a correct
> answer.

### 3. The guard holds

```
Q: What is the wifi password in the office?
  [retrieve]     attempt 1: 'What is the wifi password in the office?'
  [grade_docs]   22.4s -> '1:no 2:no 3:no'  (0/3 kept)
  [rewrite]       7.2s -> 'Wireless network access password'
  [retrieve]     attempt 2: 'Wireless network access password'
  [grade_docs]   19.3s -> '1:no 2:no 3:no'  (0/3 kept)
  [rewrite]       7.5s -> 'Network access password verification procedure'
  [retrieve]     attempt 3: 'Network access password verification procedure'
  [grade_docs]   23.9s -> '1:no 2:no 3:no'  (0/3 kept)
  [no_answer]    gave up after 3 attempts
A: I don't know based on the provided documents.
   attempts: 3     81.4s total
```

Textbook. Three genuinely different rewrites, three honest rejections, then a
clean stop. Without `MAX_ATTEMPTS` this would loop until you killed it.

---

## Honest assessment: does the loop actually help?

**On this corpus, rarely.** That is worth saying plainly rather than hiding
behind a tutorial-shaped success story.

While building this step we screened a dozen deliberately informal questions —
"Do they pay for my chair?", "How long till I'm permanent?", "What if I don't
service it?" — expecting retrieval failures to demonstrate the loop on.
**Almost all of them succeeded on attempt 1.** Two documents, 11 chunks, clean
headings: retrieval barely fails, so the loop rarely has anything to fix.

The rewrite loop earns its place when:

- the corpus is **large** (thousands of chunks — more ways to miss)
- documents use **jargon** the user doesn't share
- questions arrive **casually phrased**, as real users write them
- chunks are **messy or inconsistent** (scanned PDFs, mixed formats)

> **The lesson is not "loops are useless".** It is: add machinery when you have
> measured the problem it solves. We built it here to learn the mechanism —
> which is the right reason in a tutorial and the wrong reason in production.
> On a corpus this clean, Step 6's graph is arguably the better system: fewer
> moving parts, half the latency.

---

## Gotcha: I repeated the Step 6 import bug immediately

Step 7's first version did this:

```python
from step6_grading import GRADE_PROMPT, number_docs, parse_verdicts
```

And the output showed:

```
Building the vector store (once)...
Ready.

Building the vector store (once)...     <- twice, again
```

The *exact* bug documented one step earlier, repeated within an hour of
writing that documentation.

**Why it happened:** `prompts.py` fixed the symptom (the shared prompt) rather
than the cause — `step6_grading.py` still had expensive module-level setup
*and* definitions worth sharing. The moment Step 7 needed one of those
definitions, the bug came back.

**The proper fix** was to move every shared definition — `GRADE_PROMPT`,
`number_docs`, `parse_verdicts`, `GRADE_DOC_CHARS` — into `prompts.py`, so the
step files contain only their own graph and nothing anyone needs to import.

> **The generalisable version:** when you fix an import-side-effect bug by
> extracting *one* thing, you've treated the symptom. Ask what else lives in
> that module that someone will want later. Knowing about a bug class does not
> stop you writing it again — structure does.

---

## Gotcha: don't over-engineer the demo

The first plan for this step was a question that fails on attempt 1 and
**succeeds** after a rewrite. Finding one took many attempts:

- `"How much do I get for my desk setup?"` — failed for the wrong reason
  (a truncation bug, see below), and worked once fixed
- `"Can the robot work in a paint booth?"` — needed the model to infer that a
  paint booth is an "explosive atmosphere". Rewrites got closer but the strict
  grader still refused
- Feeding the rewriter a table of contents — made it **copy a heading
  verbatim**, producing the *same* rewrite for two unrelated questions

At some point the honest move is to stop engineering the demo and report what
actually happens. The mechanism is demonstrated perfectly by case 3; the fact
that case 2 doesn't produce a triumphant answer is *information about the
technique*, not a failure of the write-up.

---

## Gotcha (Step 6): a speed optimisation that silently broke correctness

Found while hunting for test questions, and it belongs to Step 6.

`GRADE_DOC_CHARS = 450` truncated each chunk before grading — a speed
optimisation, since prompt-reading dominates runtime.

But the home-office stipend sits at **character 724 of chunk 2**. The grader
never saw it, rejected a chunk that *did* contain the answer, and the system
confidently reported "I don't know" about a fact sitting in the file.

Measured:

```
GRADE_DOC_CHARS = 450                        GRADE_DOC_CHARS = 1200
  0/3 kept  18.0s  desk setup?                 2/3 kept  28.7s  desk setup?
  0/3 kept  18.2s  chair money?                1/3 kept  23.8s  chair money?
  0/3 kept   2.8s  crashing the arm?           1/3 kept  29.0s  crashing the arm?
  0/3 kept  17.8s  wifi password?              0/3 kept  28.1s  wifi password?  <- still correct
```

1.6x slower, and right instead of wrong. Now `1200` — larger than our biggest
chunk (975), so nothing is cut.

> **The lesson:** this optimisation had no error, no warning, and a plausible
> output. "I don't know" is exactly what a *working* system says sometimes.
> Optimise only what you have measured, and re-check correctness after —
> a faster wrong answer is not an improvement.

---

## Try it yourself

1. **Remove the guard.** Delete the `attempts >= MAX_ATTEMPTS` branch and ask
   the wifi question. It loops until LangGraph's `recursion_limit` (25) stops
   it. Watch ~25 rewrites go by. *That* is why the counter exists.

2. **Break the reducer.** Change `relevant_docs` to a plain
   `list[Document]`. Then find a question where lap 1 keeps a chunk and lap 2
   keeps none — the final answer loses lap 1's chunk entirely. Step 2's lesson,
   now with consequences.

3. **Answer the wrong question.** In `generate`, use `state["question"]`
   instead of `state["original_question"]`. Ask about the paint booth and watch
   it answer the *rewritten* query — a question the user never asked.

4. **Raise the limit.** Set `MAX_ATTEMPTS = 5` and re-ask the wifi question.
   Do the extra rewrites get better, or just different? (Mostly the latter —
   evidence that more retries is not more intelligence.)

5. **Make the loop useful.** Add a third document to `docs/` on a topic written
   in unfamiliar jargon, then ask about it casually. This is the setting where
   the loop actually pays for itself.

## ✅ Step 7 checkpoint

You should be able to answer:

- What makes an edge a "cycle", and how does it differ from a conditional edge?
- What are the three exits from our loop?
- Why must `relevant_docs` accumulate but `documents` not?
- Why keep `original_question` separate from `question`?
- Why did `generate` refuse even after the grader approved a chunk — and why
  is that good?
- When is a rewrite loop worth adding, and when is it just latency?

**Next:** Step 8 — a `route` node so casual chat skips retrieval entirely, plus
a Streamlit chat UI so this stops being a script and becomes something you can
actually use.

---

# Step 8 — `route` and a real UI

**Files:** `step8_router.py`, `app.py`
**Run:** `python step8_router.py` (CLI) · `streamlit run app.py` (web)

---

## Part 1: the `route` node

### The problem

Type "hi" into the Step 7 graph and this happens:

```
retrieve -> grade (25s) -> rewrite -> retrieve -> grade (20s) ->
rewrite -> retrieve -> grade (22s) -> no_answer
```

**~80 seconds to not answer a greeting.** The graph has one gear.

### The fix

One node up front: *does this even need the documents?*

```
START -> [classify] --- chat ---> [chat_reply] -----------> END
             |
          search
             v
        [retrieve] -> [grade_docs] -> ... (the Step 7 graph)
```

### A design point worth copying

This router runs *before* any work happens, and LangGraph would happily let you
attach a conditional edge straight to `START` and do the classification inside
the router function.

**Don't.** It breaks the Step 6 rule: nodes do work, routers only choose.
Keeping `classify` as a node means:

- you can test it alone, without running the graph
- `route_question(state)` stays a pure function you can call with a plain dict
- the classification shows up in `stream()` output, so you can see what it
  decided

The graph is now 7 nodes and 3 routers, and every router is a two-line pure
function. That is not an accident.

### Few-shot examples earned their cost

First attempt at the classifier prompt — plain instructions, no examples:

```
  OK   3.9s  want=CHAT   got=CHAT    "hi"
  OK   4.0s  want=CHAT   got=CHAT    "thanks!"
  XX   4.2s  want=CHAT   got=SEARCH  "what can you do?"     <- wrong
  ...
  8/9 correct
```

Adding six one-line examples:

```
  10/10 correct, ~7s each
```

+3 seconds of prompt-reading for perfect accuracy on the test set. Worth it
immediately: the classifier costs 7s but saves ~80s on every non-question, so
it pays for itself the first time someone types "thanks".

> **Pattern worth remembering:** the rewrite prompt in Step 7 also only worked
> once we gave it examples. A 4B model imitates demonstrated patterns far
> better than it follows abstract descriptions. When a small model won't follow
> an instruction, **show it** rather than explaining harder.

### Results

```
Q: hi there
  [classify]   -> chat
  [chat_reply] replying... 7.7s
A: Hello! I can help you with questions about handbook.md and product_faq.md...
   (chat path, 17-32s)

Q: What is the wifi password in the office?
  [classify]   -> search
  ... 3 attempts ...
A: I don't know based on the provided documents.
   (search path, 82.7s)
```

Small talk went from ~80s to ~17s. Still not fast — `classify` and
`chat_reply` are two LLM calls on a CPU. A pure-Python shortcut for exact
greetings ("hi", "hello", "thanks") would make those instant, and is a
reasonable thing to add. We left it out to keep the routing concept visible.

---

## Part 2: the module that is finally safe to import

Steps 5, 6 and 7 all built the vector store **at module level**. That bug bit
us twice. The third time, we fixed the *shape* rather than the symptom:

```python
_RESOURCES = None

def get_resources() -> dict:
    """Build the store and models once, on first use."""
    global _RESOURCES
    if _RESOURCES is None:
        store = build_vector_store()
        _RESOURCES = {"store": store, "answer_llm": get_llm(250), ...}
    return _RESOURCES
```

Nothing runs until a node actually asks for it. Verified:

```
$ python -c "import step8_router"
imported in 4.77s, no side effects        <- and that 4.7s is importing
                                             langgraph, not building anything
```

Also note `build_graph()` is now a **function** rather than module-level code.
That matters for the web app, which needs to build the graph inside a cached
function rather than at import time.

> **The general shape:** a module should define things. Doing things is the
> caller's decision.

---

## Part 3: the Streamlit app

```bash
streamlit run app.py
```

### Why the UI is more than decoration

On CPU a question takes 20-80 seconds. A terminal that prints nothing for a
minute feels broken. **Streaming the graph's progress turns dead time into
visible work** — and it doubles as the best debugging view you have, because
you watch the path the graph takes.

The whole UI is built on `stream(mode="updates")` — the same debugging tool
from Step 2, now doing double duty:

```python
for chunk in graph.stream(initial_state(question), stream_mode="updates"):
    for node, update in chunk.items():
        status.info(STEP_LABELS[node])       # live progress
        ...                                   # and build a trace
```

Each node yields as it finishes, so the user sees:

```
Deciding whether to search the documents...
Searching the documents...
Checking whether the results actually answer this...
No good match - rephrasing the question and retrying...
Writing the answer...
```

### The three things the app shows that a chatbot usually hides

1. **A "How this was answered" trace** with timings per node:

   ```
   [  7.0s] classify -> search
   [  7.7s] retrieve attempt 1: handbook.md#1, handbook.md#2, handbook.md#0
   [ 10.6s] grade -> kept 2 chunk(s)
   [ 27.8s] generate
   ```

2. **The actual source chunks** the answer was built from — not just filenames,
   the text. If the answer looks wrong you can immediately see whether the
   retrieval or the generation was at fault.

3. **The retry path**, when the loop fires. You see the rewritten query, which
   is often more informative than the answer.

Most chat UIs hide all of this. For a document assistant it is exactly what
builds (or correctly destroys) trust in an answer.

### `@st.cache_resource` — Step 5's lesson, again

```python
@st.cache_resource(show_spinner="Loading documents and models (once)...")
def load_graph():
    return build_graph(), list_sources()
```

**Streamlit re-runs the entire script on every interaction.** Without the
cache, the vector store would be rebuilt on every message. This is exactly the
"expensive setup happens once" principle from Step 5, wearing a different hat.

---

## The finished system

```
START -> [classify] --- chat ---> [chat_reply] --------------> END
             |
          search
             v
        [retrieve] <-------------------+
             |                         |
        [grade_docs] --no--> [rewrite_query]   (max 3 attempts)
             |     \
        relevant    \--- attempts exhausted --> [no_answer] --> END
             v
        [generate] -> END
```

**7 nodes. 3 routers. 1 cycle. Runs entirely offline.**

| File | What it holds |
|---|---|
| `llm.py` | model config — one line to switch to Claude |
| `prompts.py` | shared prompts + grading helpers (safe to import) |
| `step3_load_and_chunk.py` | loading and chunking |
| `step4_embed_and_store.py` | the embeddings adapter + vector store |
| `step8_router.py` | the final graph |
| `app.py` | the web UI |
| `docs/` | your documents |

Steps 2, 5, 6 and 7 remain as teaching artifacts — each is a working system at
an earlier stage, useful for comparison.

---

## Try it yourself

1. **Add your own documents.** Drop a PDF into `docs/`, restart the app. No
   code changes. This is the moment the project stops being an exercise.

2. **Watch the trace on a hard question.** Ask something oblique and open "How
   this was answered". The rewritten queries tell you what the system
   understood.

3. **Make small talk instant.** Add a plain-Python shortcut before `classify`
   for exact matches like "hi"/"hello"/"thanks". Zero LLM calls, zero seconds.
   Then measure how often it actually fires.

4. **Break the classifier.** Remove the examples from `CLASSIFY_PROMPT` and ask
   "what can you do?" It routes to `search` and spends 80 seconds failing to
   find an answer about itself.

5. **Switch to a cloud model.** Edit `get_llm()` in `llm.py` to use
   `ChatAnthropic`, add an API key, and run the same app. Nothing else changes,
   and every answer arrives in about 3 seconds. That contrast is the clearest
   demonstration of why the interface boundary mattered.

## ✅ Step 8 checkpoint

You should be able to answer:

- Why put classification in a node rather than in the router function?
- Why did few-shot examples matter more than a clearer instruction?
- What makes `step8_router.py` safe to import when Steps 5-7 were not?
- Why does `@st.cache_resource` matter in a Streamlit app specifically?
- What does `stream_mode="updates"` give you that `invoke()` does not?

---

# Step 9 — Conversation memory

**File:** `step9_memory.py` · **Run:** `python step9_memory.py`
**New dependency:** `langgraph-checkpoint-sqlite`

Two separate problems hide behind the word "memory". Solving only the first is
the classic mistake.

---

## Problem 1: the graph forgets between calls

Every `invoke()` so far started from nothing. LangGraph's fix is a
**checkpointer**: it saves state after every node, keyed by a `thread_id`.

```python
graph = builder.compile(checkpointer=saver)
graph.invoke({"user_message": "hi"}, config={"configurable": {"thread_id": "user-123"}})
```

Same `thread_id` -> state is loaded, extended, saved again.
Different `thread_id` -> a separate conversation.

Note the checkpointer is attached at **compile** time, and the thread id is
passed at **invoke** time. One compiled graph serves every conversation.

We use `SqliteSaver` so history survives a restart:

```python
conn = sqlite3.connect("memory.db", check_same_thread=False)
saver = SqliteSaver(conn)
saver.setup()
```

`check_same_thread=False` because Streamlit serves requests from a thread pool.
`InMemorySaver` also works and needs no file — but closing the app forgets
everything.

> This is the same machinery behind crash-resume and pause-for-human-input.
> Learn it here and those come almost free.

---

## Problem 2: follow-ups are useless as search queries

This is the one people miss. Add a checkpointer, feel clever, then watch this:

```
You: What is the payload of the AtlasArm A5?
Bot: 5 kg [product_faq.md]
You: what about the A12?
Bot: I don't know based on the provided documents.
```

The history was stored perfectly. **It didn't help.** `retrieve` embedded the
literal string *"what about the A12?"* — which is semantically almost nothing.
No "payload", no "robot", no "reach". The vector lands nowhere useful.

**Storing history does not fix retrieval.** The question has to be rewritten
into a self-contained one *before* retrieval ever sees it:

```
"what about the A12?"  ->  "What is the payload of the AtlasArm A12?"
```

That's the `contextualize` node.

### Two rewriters, two different jobs

Easy to confuse, worth separating clearly:

| Node | Rewrites | Because | Runs |
|---|---|---|---|
| `contextualize` (Step 9) | pronouns and ellipsis, against history | the question isn't self-contained | every turn with history |
| `rewrite_query` (Step 7) | vocabulary, against the corpus | the search *failed* | only after 0 chunks kept |

Both produce a new query string. They are not interchangeable.

### The optimisation that matters

`contextualize` skips the LLM entirely on the first turn — there is no history
to resolve, so there is nothing to do. That saves ~15s on the first question of
every conversation:

```python
if not history:
    standalone = user_msg
    print("  [contextualize] first turn, no rewrite needed")
```

---

## The trap: reducers + a checkpointer leak state across turns

This is the subtle one, and it comes straight back to Step 2.

`relevant_docs` uses a reducer so the retry loop can **accumulate** across laps
(Step 7). Within one question that's exactly right.

**But a checkpointer persists state across questions too.** So question 2 would
append to question 1's documents, question 3 to both, and the context would
quietly fill with chunks from unrelated questions. `attempts` is worse — it
would arrive at question 2 already set to `3`, and the retry loop would never
run again.

So per-turn fields must be **explicitly reset** at the start of every turn. And
here's the catch: **you cannot reset an accumulating field by passing `[]`** —
the reducer just appends an empty list. You need a reducer that understands a
"clear" sentinel:

```python
def append_or_reset(old: list, new) -> list:
    """Append normally; treat None as 'clear this field'."""
    if new is None:
        return []
    return (old or []) + new
```

```python
relevant_docs: Annotated[list[Document], append_or_reset]
tried_queries: Annotated[list[str], append_or_reset]
```

And `contextualize` — which starts every turn — does the reset:

```python
return {
    "messages": [HumanMessage(content=user_msg)],
    "standalone_question": standalone,
    "question": standalone,
    # --- reset per-turn scratch, or it leaks across the conversation ---
    "relevant_docs": None,      # sentinel -> append_or_reset clears it
    "tried_queries": None,
    "documents": [],
    "attempts": 0,
    "answer": "",
}
```

> **The general principle:** with a checkpointer, state divides into two kinds
> — **conversation state** that must persist (`messages`) and **scratch state**
> that must not (`attempts`, `documents`, `relevant_docs`). Decide which is
> which for every field, and reset the scratch explicitly. A field that
> accumulates *and* persists is a slow-growing bug.

---

## Three names for "the question"

Memory forces a distinction we could previously ignore:

| Field | Meaning | Used by |
|---|---|---|
| `user_message` | exactly what the user typed | history, display |
| `standalone_question` | the same question, self-contained | `classify`, `generate`, `rewrite_query` |
| `question` | the current *search query*, rewritten by the retry loop | `retrieve`, `grade_docs` |

Collapsing any two of these breaks something. If `generate` answered
`question`, it would answer whatever the retry loop last invented. If
`retrieve` searched `user_message`, follow-ups would fail.

`messages` uses LangGraph's purpose-built `add_messages` reducer, which appends
and handles message ids properly — the chat-history equivalent of
`operator.add`.

---

## The output

```
You: What is the payload of the AtlasArm A5?
  [contextualize] first turn, no rewrite needed
  [classify]      -> search
  [retrieve]      attempt 1: 'What is the payload of the AtlasArm A5?'
  [grade_docs]    34.4s -> '1:yes 2:no 3:yes' (2/3 kept)
Bot: The AtlasArm A5 has a maximum payload of 5 kg [product_faq.md #6].
     (97.7s, 2 messages in history)

You: what about the A12?
  [contextualize] 13.9s -> 'What is the payload of the AtlasArm A12?'
  [classify]      -> search
  [retrieve]      attempt 1: 'What is the payload of the AtlasArm A12?'
Bot: The AtlasArm A12 has a maximum payload of 12 kg [product_faq.md #6].
     (77.7s, 4 messages in history)

You: and how far can it reach?
  [contextualize] 16.4s -> 'And how far can the AtlasArm A12 reach?'
Bot: The AtlasArm A12 has a reach of 1,300 mm [product_faq.md #6].
     (76.1s, 6 messages in history)

You: thanks!
  [contextualize] 17.7s -> 'Thank you!'
  [classify]      -> chat
Bot: You're welcome! Is there anything else I can help you with regarding
     the AtlasArm models?
     (chat path, 40.4s, 8 messages in history)
```

**Turn 3 is the one to notice.** "and how far can it reach?" became "And how far
can the AtlasArm **A12** reach?" — the model carried "A12" forward from *two
turns earlier*, through a question that never mentioned it. That is the
contextualiser reading actual conversation history, not just the previous line.

### It survives a restart

A brand-new process, reading `memory.db`:

```
resuming thread_id=demo-45f11ea9 in a fresh process

recovered from disk:
  Human     What is the payload of the AtlasArm A5?
  AI        The AtlasArm A5 has a maximum payload of 5 kg [product_faq.md #6
  Human     what about the A12?
  AI        The AtlasArm A12 has a maximum payload of 12 kg [product_faq.md
  Human     and how far can it reach?
  AI        The AtlasArm A12 has a reach of 1,300 mm [product_faq.md #6]. Th
  Human     thanks!
  AI        You're welcome! Is there anything else I can help you with regar
```

`graph.get_state(config)` reads a thread's state without running anything —
useful for debugging, and for rendering history when a UI reloads.

---

## The cost

Memory is not free: `contextualize` adds an LLM call to every turn after the
first, ~15s on this hardware. Cheaper options exist if that hurts:

- **Skip it when the question is already long and specific** — a crude
  heuristic (no pronouns, more than 6 words) catches most standalone questions
  with zero LLM calls
- **Use a smaller model for this node only** — resolving a pronoun is far
  easier than grading relevance, and `gemma3:1b` may well handle it
- **Only contextualise when `classify` says "search"** — small talk never needs
  a resolved question

We kept it simple and unconditional so the mechanism stays visible.

---

## Try it yourself

1. **Prove problem 2 exists.** In `contextualize`, always return
   `standalone = user_msg` (skip the rewrite). Ask the A5 question, then "what
   about the A12?" You'll get "I don't know" — with the history stored
   perfectly. That is the whole lesson in one experiment.

2. **Watch state leak.** Change `relevant_docs` back to `operator.add` and
   remove the reset. Ask three different questions in a row and print
   `len(state["relevant_docs"])` in `generate`. It grows: 2, 4, 7... Answers get
   worse as unrelated chunks pile into the context.

3. **Freeze the loop.** Remove `"attempts": 0` from the reset. Ask an
   unanswerable question (3 attempts), then any other question — it goes
   straight to `no_answer`, because `attempts` is still 3.

4. **Resume a conversation.** Note the `thread_id` printed by the demo, then in
   a new process call `graph.get_state(thread(tid))` — or just `invoke` another
   message on the same id and watch it continue.

5. **Two conversations at once.** Invoke with `thread_id="a"` and
   `thread_id="b"` alternately. They never see each other's history. This is
   how one server handles many users.

## ✅ Step 9 checkpoint

You should be able to answer:

- What does a checkpointer save, and what selects *which* conversation?
- Why doesn't storing chat history fix follow-up questions?
- How do `contextualize` and `rewrite_query` differ?
- Why can't you reset an accumulating field by passing `[]`?
- Which state fields must persist across turns, and which must be reset?
- Why keep `user_message`, `standalone_question` and `question` separate?

---

# Step 10 — Trace logging

**Files:** `trace_log.py`, `analyze_logs.py`
**Run:** `python step9_memory.py`, then `python analyze_logs.py`

---

## Why `print()` wasn't enough

The nodes had `print()` calls from Step 5 onward. That is fine for *watching* a
run and useless for *reviewing* one:

- it scrolls away
- it isn't timestamped
- it drops exactly the details you need when an answer looks wrong — the full
  chunk text, the similarity scores, the grader's raw output

So: two outputs from one system.

| File | Answers the question |
|---|---|
| `logs/assistant.log` | *"What happened on that one question?"* |
| `logs/questions.jsonl` | *"How is the system doing overall?"* |

---

## Log levels do the work

The console should stay readable; the file should hold everything. That is
precisely what log **levels** are for, and it means one call site serves both:

```python
logger.info(f"  results        : {summary}")     # console + file
logger.debug(f"        | {line}")                # file only - full chunk text
```

```python
fh = logging.FileHandler(TEXT_LOG, encoding="utf-8")
fh.setLevel(logging.DEBUG)      # everything

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)       # the short version
```

Two details worth stealing:

- **`encoding="utf-8"` on the file handler.** Without it the log file inherits
  the Windows ANSI codepage and crashes on the first em-dash.
- **`StreamHandler(sys.stdout)`, not the default `sys.stderr`.** We hide library
  warnings with `2>/dev/null`, and the default would have hidden the whole
  trace along with them. It did, for one run, and looked like the logger was
  broken.

---

## How nodes reach the current trace

This is the interesting design problem. A node needs to write to "the log for
the question currently being answered", but:

- **It can't go in graph state.** The SQLite checkpointer would try to
  serialise a logger object.
- **It can't be a module-level global.** Streamlit serves requests from a
  thread pool, so two users would overwrite each other's trace.

The right tool is **`contextvars`** — a value visible to everything in the
current execution context, and separate per thread:

```python
_current: contextvars.ContextVar = contextvars.ContextVar("trace", default=None)

def start_trace(user_message, thread_id) -> QuestionTrace:
    t = QuestionTrace(user_message, thread_id)
    _current.set(t)
    return t

def current():
    return _current.get() or _NULL
```

### The null object

`current()` never returns `None`. When no trace is active it returns a
`NullTrace` whose every method does nothing:

```python
class NullTrace:
    def __getattr__(self, _name):
        def noop(*args, **kwargs):
            return None
        return noop
```

So nodes call `trace().grade(...)` unconditionally — no `if trace is not None`
scattered through seven functions. Anyone can still `import step9_memory` and
call the graph without logging, and nothing breaks.

> **The null object pattern:** when an optional collaborator is used in many
> places, a do-nothing implementation is usually cleaner than a null check at
> every call site.

---

## What a question looks like

```
====================================================================================
QUESTION #2  |  thread=demo-9a549372  |  2026-09-03T16:14:51
====================================================================================
USER MESSAGE : what about the A12?

[contextualize]  14.3s  (t+14.3s)
  standalone     : 'What is the payload of the AtlasArm A12?'

[classify]  7.7s  (t+22.0s)
  route          : search

[retrieve]  0.3s  (t+22.4s)  attempt 1
  query          : 'What is the payload of the AtlasArm A12?'
  results        : product_faq.md#6(0.832), product_faq.md#8(0.703), product_faq.md#7(0.693)

[grade_docs]  22.6s  (t+45.0s)  attempt 1
  raw output     : '1:yes 2:no 3:yes'
  kept           : 2/3
    [1] product_faq.md#6  -> KEEP
    [2] product_faq.md#8  -> drop
    [3] product_faq.md#7  -> KEEP

[generate]  33.2s  (t+78.2s)
  chunks used    : product_faq.md#6, product_faq.md#7
  tokens         : in=638 out=71
------------------------------------------------------------------------------------
RESULT   route=search  attempts=1  total=78.2s
ANSWER   The AtlasArm A12 has a maximum payload of 12 kg [product_faq.md #7]. ...
====================================================================================
```

Each node line carries **two** times: how long that node took, and `t+` since
the question started. Both matter — the first tells you what to optimise, the
second tells you what the user is waiting through.

Retrieval now uses `similarity_search_with_score` instead of
`similarity_search`, purely so the log can show *how close* each chunk was, not
just which ones came back. When retrieval looks wrong, `0.85` versus `0.51`
tells you whether it was confident or scraping.

The **file** additionally holds the full text of every retrieved chunk. That is
the single most useful thing in it: when an answer is wrong you can read
exactly what the model was looking at.

---

## Analysing across questions

```bash
python analyze_logs.py
```

```
4 questions logged

TIME BY NODE  (total across all questions)
  grade_docs        108.5s  over   5 call(s)   avg  21.7s
  generate           62.9s  over   2 call(s)   avg  31.4s
  contextualize      47.5s  over   4 call(s)   avg  11.9s
  classify           29.5s  over   4 call(s)   avg   7.4s
  rewrite_query      14.5s  over   2 call(s)   avg   7.3s
  retrieve            1.6s  over   5 call(s)   avg   0.3s

RETRY LOOP
  fired on 1/4 questions
    3 attempts, gave up    'What is the wifi password in the office?'

GRADING
  5 grading call(s), 3 kept nothing (60%)
```

**That table immediately reprices the whole system.** Before seeing it, the
instinct is that "the search" is the expensive part. It isn't:

- `retrieve` — the actual vector search — is **1.6 seconds across every
  question asked**. Effectively free.
- `grade_docs` is **the most expensive node in the system**, at 108s.

So if you wanted this faster, you'd attack grading — a smaller model for that
node, or fewer chunks graded — and you would never touch the vector store.
That is the entire argument for structured logging in one screen.

> This is the same lesson as the Step 5 timing correction, arriving by a better
> route. There we guessed wrong and had to be corrected by measurement. Here
> the measurement is simply always on.

---

## Reading the log to debug an answer

| Symptom | What to look for |
|---|---|
| "I don't know" about something in your docs | Did `retrieve` find the right chunk? Check the scores. If yes, the grader dropped it — read `raw output` |
| Wrong or invented answer | Look at `chunks used`, then read that chunk's text in the log. Every claim should be traceable to it |
| Follow-up misunderstood | Check `contextualize` — what did it rewrite the question into? |
| Too slow | `python analyze_logs.py`, look at TIME BY NODE |

This is the concrete version of the Step 7 lesson about telling **Case A** (the
grader was wrong) from **Case B** (the answerer was wrong). Before, that
required adding a debug print. Now it's in the log by default.

---

## Try it yourself

1. **Ask something wrong on purpose.** Ask about a fact that is in your docs
   but worded oddly. Open `logs/assistant.log`, find the question, and work out
   whether retrieval or grading failed. This is the actual daily workflow.

2. **Watch the log grow across a session.** Run `analyze_logs.py` after 5
   questions, then after 20. The "GRADING kept nothing" percentage is the most
   useful number in it — if it's high, your grader is too strict.

3. **Log something new.** Add the similarity score of the *best rejected* chunk
   to the grade event. If good answers and bad answers separate cleanly on that
   number, you've found a cheap pre-filter that could skip the grading LLM
   entirely.

4. **Turn the console quiet.** Set the StreamHandler level to `WARNING`. The
   file still records everything; the terminal goes silent. That is how you'd
   run it in production.

## ✅ Step 10 checkpoint

You should be able to answer:

- Why log to a file *and* the console, and how do levels make that one call?
- Why can't the trace live in graph state, or in a module-level global?
- What does the null-object `NullTrace` save you from writing?
- Which node is most expensive, and how do you know?
- Given "I don't know" about a fact in your documents, what do you check first?

---
