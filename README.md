# 📄 Agentic RAG Assistant

A **"chat with my documents"** assistant built with **LangGraph**, running
**entirely on your machine** — no API key, no cost, nothing leaves your computer.

Ask questions about your PDFs or notes. It searches them, **verifies the results
actually answer your question**, retries with better wording when they don't,
answers with citations — or honestly says it doesn't know. Follow-up questions
work.

> Built step by step as a beginner's LangGraph tutorial.
> The full teaching notes — including every mistake made along the way — are in
> **[TUTORIAL.md](TUTORIAL.md)**.

---

## Table of contents

- [What it looks like](#what-it-looks-like)
- [The flow](#the-flow)
- [The 10 steps](#the-10-steps)
- [Installation](#installation)
- [How to run it](#how-to-run-it)
- [Using your own documents](#using-your-own-documents)
- [The trace log](#the-trace-log)
- [Project layout](#project-layout)
- [Performance](#performance)
- [Switching to a cloud model](#switching-to-a-cloud-model)
- [Troubleshooting](#troubleshooting)
- [Deploying it live (free)](#deploying-it-live-free)
- [Known limitations](#known-limitations)

---

## What it looks like

```
You: What is the payload of the AtlasArm A5?
Bot: The AtlasArm A5 has a maximum payload of 5 kg [product_faq.md #6].

You: what about the A12?                    <- a follow-up, not a full question
Bot: The AtlasArm A12 has a maximum payload of 12 kg [product_faq.md #6].

You: and how far can it reach?              <- "it" still means the A12
Bot: The AtlasArm A12 has a reach of 1,300 mm [product_faq.md #6].

You: What is the wifi password?             <- genuinely not in the documents
Bot: I don't know based on the provided documents.
```

That last line is the point. Most RAG demos would invent something.

---

## The flow

```
                         user message
                              |
                              v
                    +-------------------+
                    |  contextualize    |   resolve follow-ups against history
                    +-------------------+   "what about the A12?"
                              |               -> "What is the payload of the A12?"
                              v
                    +-------------------+
                    |     classify      |   does this even need the documents?
                    +-------------------+
                        |            |
                     chat          search
                        |            |
                        v            v
              +--------------+  +-----------+
              |  chat_reply  |  | retrieve  | <-------------------+
              +--------------+  +-----------+                     |
                        |            |                            |
                        |            v                            |
                        |     +--------------+                    |
                        |     |  grade_docs  |  do these chunks    |
                        |     +--------------+  actually answer it?|
                        |        |    |    |                       |
                        |    yes |    | no |                       |
                        |        |    |    +---> +---------------+ |
                        |        |    |          | rewrite_query |-+
                        |        |    |          +---------------+
                        |        |    |            (max 3 attempts)
                        |        |    |
                        |        |    +-- attempts exhausted --+
                        |        v                             v
                        |  +------------+              +-------------+
                        |  |  generate  |              |  no_answer  |
                        |  +------------+              +-------------+
                        |        |                            |
                        +--------+----------------------------+
                                 |
                                 v
                                END
```

**8 nodes · 3 routers · 1 cycle · conversation memory**

### Why each node exists

A plain RAG pipeline is just `retrieve → answer`. That fails in four specific
ways, and each node here fixes exactly one:

| Failure of plain RAG | Node that fixes it |
|---|---|
| "what about the A12?" retrieves nothing — it's meaningless as a search query | **contextualize** |
| Typing "hi" burns 80 seconds searching documents | **classify** |
| Answers confidently from chunks that don't contain the answer | **grade_docs** |
| Gives up when your wording differs from the document's wording | **rewrite_query** |

The **cycle** (`rewrite_query → retrieve`) is the part a normal chain physically
cannot do: the graph goes *backwards* and tries again.

---

## The 10 steps

The project was built one concept at a time. Each step's file still runs on its
own, so you can watch the system grow. Full explanations in
**[TUTORIAL.md](TUTORIAL.md)**.

| # | Step | File | What it teaches |
|---|------|------|-----------------|
| 0 | Concepts | — | What RAG is; what a graph adds over a chain |
| 1 | Project setup | `requirements.txt` | venv, dependencies, why each one |
| 2 | Hello Graph | `step2_hello_graph.py` | **State, nodes, edges, reducers** — 3 fake nodes, no LLM |
| 3 | Load & chunk | `step3_load_and_chunk.py` | Why chunk size and overlap decide everything |
| 4 | Embed & store | `step4_embed_and_store.py` | Embeddings, cosine similarity, vector search |
| 5 | First working RAG | `step5_first_rag.py` | `retrieve → generate`, prompt anatomy, citations |
| 6 | Grading | `step6_grading.py` | **Conditional edges** — the graph starts deciding |
| 7 | The retry loop | `step7_rewrite_loop.py` | **Cycles** — an edge pointing backwards, and its guard |
| 8 | Routing + UI | `step8_router.py`, `app.py` | Skip the docs for small talk; a real interface |
| 9 | Memory | `step9_memory.py` | **Checkpointers**, `thread_id`, follow-up questions |
| 10 | Trace logging | `trace_log.py`, `analyze_logs.py` | contextvars, log levels, measuring where time goes |

**Recommended reading order:** run `step2` first — it has no LLM, no documents,
and no magic, and it's where the core ideas actually land. Then read `step4`
(the concept at the heart of RAG) and `step6` (the concept at the heart of
LangGraph).

---

## Installation

### Prerequisites

**1. Python 3.11 or newer**

```bash
python --version
```

**2. [Ollama](https://ollama.com)** — runs the language model locally.
Download the installer, then pull the model:

```bash
ollama pull gemma3:4b
```

That's a 3.3 GB download. Verify it's running:

```bash
ollama list
```

> **Why `gemma3:4b`?** We benchmarked the alternatives. `gemma3:1b` is faster
> but **failed the relevance-grading task** — it called a clearly relevant
> document irrelevant, which breaks the whole design. Larger models won't fit
> comfortably in RAM alongside everything else. If you have a GPU, try a 7B or
> 8B model and edit `MODEL` in `llm.py`.

### Set up the project

```bash
# 1. Create an isolated environment for this project's packages
python -m venv .venv

# 2. Activate it
.venv\Scripts\activate            # Windows PowerShell
# source .venv/bin/activate       # macOS / Linux

# Your prompt should now start with (.venv)

# 3. Install everything
pip install -r requirements.txt
```

**First run only:** a ~50 MB embedding model downloads automatically into
`.model_cache/`. After that the project works with no internet connection at all.

---

## How to run it

### The web app (recommended)

```bash
streamlit run app.py
```

Opens at <http://localhost:8501>.

The interface shows three things most chat apps hide:

- **Live progress** — which node is running right now, so a 40-second answer
  doesn't look like a hang
- **Sources** — the actual chunk text each answer was built from, not just
  filenames
- **"How this was answered"** — a timed trace of every node, including any
  rewritten queries

### The command line

```bash
python step9_memory.py       # the finished system, with a scripted conversation
python step8_router.py       # without memory
python step7_rewrite_loop.py # without routing
python step6_grading.py      # without the retry loop
python step5_first_rag.py    # plain RAG
python step4_embed_and_store.py   # search only, no LLM
python step3_load_and_chunk.py    # chunking only
python step2_hello_graph.py       # graph mechanics only, instant
```

Running these in reverse order (2 → 9) is the guided tour.

---

## Using your own documents

1. Drop `.md`, `.txt` or `.pdf` files into `docs/`
2. Restart the app

That's the whole process — no code changes, no re-indexing command.

You can delete the three sample files (`handbook.md`, `product_faq.md`,
`it_policy.pdf`) once you have your own.

### How PDFs are handled

PDFs are loaded **one Document per page**, so citations point at a page rather
than at a whole file:

```
ANSWER  The primary office wireless network uses the password
        Ferrite-Anchor-88 [it_policy.pdf p.1 #6]. This password is
        rotated every 90 days [it_policy.pdf p.1 #6].
```

Blank pages are skipped automatically.

### ⚠️ Scanned PDFs will not work

A scanned PDF is a **picture** of text, not text. `pypdf` extracts nothing from
it, and the failure is **silent** — the file loads, no error appears, and the
assistant simply never knows anything about it.

The loader warns you when this happens:

```
WARNING: contract.pdf has 12 page(s) but no extractable text.
         It is probably a SCANNED pdf (an image). You need OCR - e.g.
         ocrmypdf - to use it.
```

To fix, run OCR on the file first:

```bash
pip install ocrmypdf          # also needs Tesseract installed
ocrmypdf scanned.pdf searchable.pdf
```

**How to tell before you start:** open the PDF and try to select a sentence
with your mouse. If you can't select the text, neither can the loader.

> **Tip:** if answers come back poor, the usual cause is chunking, not the
> model. Try `CHUNK_SIZE = 500` for dense reference material or `1500` for
> flowing prose, in `step3_load_and_chunk.py`.

---

## The trace log

Every question writes a complete record of what happened. Two files:

| File | What it's for |
|---|---|
| `logs/assistant.log` | Human-readable, full detail. Answers *"what happened on that one question?"* |
| `logs/questions.jsonl` | One JSON object per question. Answers *"how is the system doing overall?"* |

### What a question looks like in the log

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

The **file** contains more than the console shows — the full text of every
retrieved chunk, the raw model output at each step, and read/write token
timings. That is the DEBUG level, and it's what you actually need when an
answer looks wrong.

### Analysing many questions

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
  retrieve            1.6s  over   5 call(s)   avg   0.3s

RETRY LOOP
  fired on 1/4 questions
    3 attempts, gave up    'What is the wifi password in the office?'

GRADING
  5 grading call(s), 3 kept nothing (60%)
```

That output immediately tells you where to spend effort: **grading is the most
expensive node in the system**, and retrieval is essentially free. If you wanted
this faster, you'd attack `grade_docs` — not the vector search.

### Reading the log to debug an answer

| Symptom | What to look for |
|---|---|
| "I don't know" about something in your docs | Did `retrieve` find the right chunk? Check the scores. If yes, the grader dropped it — look at `raw output` |
| Wrong or made-up answer | Look at `chunks used` and read that chunk's text in the log. The answer should be traceable to it |
| Follow-up misunderstood | Check `contextualize` — what did it rewrite the question to? |
| Too slow | `python analyze_logs.py` and look at TIME BY NODE |

---

## Project layout

```
rag_assistant/
├── app.py                      # Streamlit web UI
├── step9_memory.py             # THE FINAL GRAPH - start here to read the code
│
├── llm.py                      # model config — the only file that names a provider
├── prompts.py                  # shared prompts + grading helpers
├── step3_load_and_chunk.py     # load files -> chunks
├── step4_embed_and_store.py    # embeddings adapter + vector store
│
├── step2_hello_graph.py        # teaching artifacts: each is a working system
├── step5_first_rag.py          # at an earlier stage, kept for comparison
├── step6_grading.py
├── step7_rewrite_loop.py
├── step8_router.py
│
├── trace_log.py                # per-question logging
├── analyze_logs.py             # summarise logs/questions.jsonl
│
├── docs/                       # YOUR DOCUMENTS GO HERE
├── logs/
│   ├── assistant.log           # full human-readable trace, appended forever
│   └── questions.jsonl         # one JSON object per question
├── memory.db                   # conversation history (created on first run)
├── .model_cache/               # downloaded embedding model
├── requirements.txt
├── README.md                   # this file
└── TUTORIAL.md                 # the full teaching notes, 9 steps
```

---

## Performance

Measured on an **i7-10510U** (4 cores, **no usable GPU**) with `gemma3:4b`:

| What you do | Time |
|---|---|
| Small talk ("hi", "thanks") | ~20–40s |
| A question it can answer | ~40–90s |
| A follow-up question | +15s (the extra contextualize call) |
| A question it can't answer (3 attempts, then gives up) | ~80s |

CPU inference is the entire bottleneck: roughly **26 tokens/sec reading** a
prompt and **6 tokens/sec writing** an answer. A GPU changes this completely,
and a cloud model makes it ~3 seconds.

Two things that genuinely help on CPU:

```powershell
ollama ps                  # is another model hogging RAM?
ollama stop <other-model>  # free it — this alone was a 3x speedup for us
```

`num_thread=8` is already set in `llm.py` (Ollama defaults to physical cores;
using all 8 logical threads measured 35% faster here).

---

## Choosing a language model

The app **auto-detects** its provider from whichever API key is present, so the
same code runs offline on your laptop and on a hosted server.

| Set this env var / secret | Provider | Model | Cost | Speed |
|---|---|---|---|---|
| *(nothing)* | **Ollama** (local) | `gemma3:4b` | free | ~40-90s per question on CPU |
| `GROQ_API_KEY` | **Groq** | `llama-3.3-70b-versatile` | **free tier, no card** | ~2-5s |
| `GOOGLE_API_KEY` | **Gemini** | `gemini-2.5-flash` | **free tier** | ~3-6s |
| `ANTHROPIC_API_KEY` | **Claude** | `claude-opus-5` | paid, no free tier | ~5-10s |

Force one with `LLM_PROVIDER=groq|gemini|anthropic|ollama`.
Change any model in the `MODELS` dict at the top of **`llm.py`** - that file is
the only one that names a provider. Nothing in the graph changes.

**A hosted model is not just faster - it is more accurate.** Relevance grading
is a judgment task, and `gemma3:4b` is small for it; in one measured run it
approved an irrelevant chunk *and* rejected the one holding the answer. A 70B
model on Groq gets these right far more often.

---

## Deploying it live (free)

Streamlit Community Cloud hosts this for free. Total time: about 10 minutes.

> **Why not Vercel?** Vercel runs short-lived serverless functions. Streamlit
> needs a persistent WebSocket server, our dependencies are ~490 MB against
> Vercel's 250 MB limit, and a question can take longer than its 60s function
> ceiling. It is the wrong shape of host for this app.

### 1. Get a free API key

The deployed app can't run Ollama, so it needs a hosted model. **Groq** is the
easiest - free, no credit card:

1. Go to <https://console.groq.com/keys>
2. Sign in with Google or GitHub
3. **Create API Key**, copy it (starts with `gsk_`)

*(Google AI Studio at <https://aistudio.google.com/apikey> works the same way if
you prefer Gemini.)*

### 2. Try it locally first

```powershell
$env:GROQ_API_KEY = "gsk_your_key_here"
streamlit run app.py
```

The sidebar should read `Language model: groq / llama-3.3-70b-versatile`, and
answers should arrive in seconds rather than a minute. Fix any problems here,
where the feedback loop is fast.

### 3. Push to GitHub

```bash
git init
git add .
git commit -m "Agentic RAG assistant"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/rag-assistant.git
git push -u origin main
```

`.gitignore` already excludes your key, `memory.db`, `logs/`, `.venv/` and the
model cache. The three sample documents in `docs/` **are** committed, so the
deployed app has something to answer questions about.

> ⚠️ **Check before pushing:** `git status` should not list `.env` or
> `.streamlit/secrets.toml`. If you put private documents in `docs/`, add them
> to `.gitignore` first - a public repo is public.

### 4. Deploy

1. Go to <https://share.streamlit.io> and sign in with GitHub
2. **Create app** -> pick your repo, branch `main`, main file `app.py`
3. Open **Advanced settings -> Secrets** and paste:

   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   CHECKPOINT_DB = "/tmp/memory.db"
   ```

4. **Deploy**

First boot takes 2-3 minutes (installing packages, downloading the 50 MB
embedding model). After that you have a public URL.

### 5. What to expect

- **Conversation memory resets when the app sleeps.** Streamlit Cloud's disk is
  ephemeral. `make_checkpointer()` already falls back to in-memory storage if
  the disk is read-only, so nothing crashes. For durable memory, point
  `CHECKPOINT_DB` at a mounted volume or swap in a Postgres checkpointer.
- **The app sleeps after ~7 days idle** on the free tier; any visitor wakes it.
- **Documents are whatever is in `docs/` in the repo.** Visitors cannot upload
  files - adding an uploader is a natural next feature.
- **Anyone with the link can use it**, spending your free API quota. Streamlit
  Cloud has a private-app option if that matters.

### Other hosts

| Host | Cost | Good for |
|---|---|---|
| **Streamlit Community Cloud** | free | this app, exactly as-is |
| **Render / Railway** | ~$7/mo | persistent disk, so memory survives restarts |
| **Hugging Face Spaces** | free | an ML-audience demo |
| **ngrok / Cloudflare Tunnel** | free | sharing your *local* copy for an hour, Ollama and all - zero code changes |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | The venv isn't active. Run `.venv\Scripts\activate` — your prompt should show `(.venv)` |
| `.venv\Scripts\activate` is blocked | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or skip activation and use `.venv\Scripts\python.exe app.py` |
| Connection refused on port 11434 | Ollama isn't running. Start the Ollama app, or run `ollama serve` |
| Everything is 3x slower than the table above | `ollama ps` — another model is probably still loaded. `ollama stop <name>` |
| Garbled characters like `Northwind ? Robotics` | Windows console encoding (cp1252). Display-only; your data is fine. Use `set PYTHONUTF8=1` if it bothers you |
| Answers "I don't know" about something that IS in your docs | Open `logs/assistant.log` and find the question. If `retrieve` found the chunk but `grade_docs` dropped it, loosen `GRADE_PROMPT` in `prompts.py`. If retrieval missed it, raise `K` |
| First run hangs for a minute | It's downloading the 50 MB embedding model. Once only |
| Deployed app says it can't reach Ollama | The key isn't set. Check the sidebar - it should name groq/gemini, not ollama. Re-check the Secrets in Streamlit Cloud |
| `RateLimitError` on Groq | Free tier limit hit. Wait a minute, or switch to `GOOGLE_API_KEY` |
| Memory resets on the deployed app | Expected - Streamlit Cloud's disk is ephemeral. See step 5 of the deploy guide |

---

## Stack

- **[LangGraph](https://langchain-ai.github.io/langgraph/)** — the state machine
- **[fastembed](https://github.com/qdrant/fastembed)** (`bge-small-en-v1.5`) —
  local embeddings, 384 dimensions, no PyTorch
- **Language model** — pluggable: Groq, Gemini, Claude, or local Ollama (`llm.py`)
- **`InMemoryVectorStore`** — brute-force search; swap for Chroma when you
  outgrow it
- **SQLite checkpointer** — conversation memory that survives restarts
- **Streamlit** — the UI

---

## Known limitations

These are measured, not hypothetical:

- **Relevance grading is imperfect.** `gemma3:4b` is small for a judgment task.
  In one measured run it approved an irrelevant chunk *and* rejected the one
  holding the answer. A larger model grades noticeably better — this is where
  model size shows most.
- **Strict grading costs recall.** Every filter that prevents a wrong answer
  also occasionally discards a right one. Tune `GRADE_PROMPT` toward your use
  case: strict for policy or compliance, loose for exploration.
- **The rewrite loop rarely helps on small, tidy corpora.** On 11 chunks with
  clean headings, retrieval barely fails. It earns its place on large or
  jargon-heavy document sets.
- **No hybrid search.** Pure vector search misses exact identifiers — part
  numbers, error codes, names. Production systems combine it with keyword
  (BM25) search.
- **The vector store is rebuilt on every startup.** Fine for a few documents,
  slow past a few hundred. Swap `InMemoryVectorStore` for Chroma to persist it.
