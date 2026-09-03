"""
Step 4 - Embed & store
======================
Turn text chunks into vectors, put them in a searchable store, and search
by MEANING instead of by keyword.

Still no API key. The embedding model runs on your machine.
First run downloads ~50 MB; after that it's offline and instant.

Run it:   python step4_embed_and_store.py
"""

import math
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from step3_load_and_chunk import chunk_documents, load_documents

# Keep the downloaded model inside the project so it persists between runs.
CACHE_DIR = str(Path(__file__).parent / ".model_cache")

# 384 dimensions, ~50 MB, runs on CPU in milliseconds. A good default.
MODEL_NAME = "BAAI/bge-small-en-v1.5"


# ---------------------------------------------------------------------------
# 1. THE ADAPTER
#
# Any "embeddings object" in LangChain is just these two methods. Implement
# them and every vector store in LangChain will accept your object.
# ---------------------------------------------------------------------------
class FastEmbedAdapter(Embeddings):
    """Plugs the `fastembed` library into LangChain's Embeddings interface."""

    def __init__(self, model_name: str = MODEL_NAME, cache_dir: str = CACHE_DIR):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed chunks that go INTO the store."""
        # fastembed returns numpy arrays; LangChain wants plain lists.
        return [vec.tolist() for vec in self.model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        """Embed a QUESTION. A separate method on purpose - see note below.

        WHY TWO METHODS? Some embedding models are trained ASYMMETRICALLY:
        documents are embedded plainly, but queries get a hidden instruction
        prefix ("Represent this sentence for searching relevant passages:")
        that nudges a question toward the statements that ANSWER it rather
        than toward other questions. The interface has two methods so a model
        that needs this can do it.

        MEASURED REALITY for this model: fastembed's query_embed() returns a
        vector IDENTICAL to embed_documents() for bge-small-en-v1.5 - the
        prefix is not applied. We tested adding it by hand; it moved scores
        by about 0.01 and changed no rankings that mattered. So we keep this
        simple. Verify, don't assume.
        """
        return list(self.model.query_embed(text))[0].tolist()


# ---------------------------------------------------------------------------
# 2. BUILD THE STORE  (Step 5 imports this)
# ---------------------------------------------------------------------------
def build_vector_store() -> InMemoryVectorStore:
    """Load docs -> chunk them -> embed them -> return a searchable store."""
    docs = load_documents()
    chunks = chunk_documents(docs)
    embeddings = FastEmbedAdapter()

    # from_documents() calls embed_documents() on every chunk and keeps the
    # vector alongside the text and metadata.
    return InMemoryVectorStore.from_documents(chunks, embedding=embeddings)


# ---------------------------------------------------------------------------
# 3. HELPERS FOR THE DEMO
# ---------------------------------------------------------------------------
def cosine_similarity(a: list[float], b: list[float]) -> float:
    """The angle between two vectors, ignoring their length.

    1.0 = same direction (same meaning), 0.0 = perpendicular (unrelated).
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def preview(text: str, width: int = 60) -> str:
    return text.replace("\n", " ").strip()[:width] + "..."


def describe(doc: Document) -> str:
    """Describe a chunk by BOTH its opening text and the headings inside it.

    IMPORTANT LESSON, learned the hard way while writing this:

      * Preview the first 60 chars only -> misleading. Chunk 4 opens with
        "Client entertainment is reimbursable..." but also contains the whole
        Equipment & Security section. It looked like a bad match for
        "laptop stolen" when it was actually a good one.

      * List the headings only -> ALSO misleading. Chunk 5 opens with the
        "Lost or stolen equipment must be reported..." paragraph BEFORE its
        first heading, so heading-only display hides the very text that
        matched.

    A chunk spans section boundaries. Show both ends or you will misread
    your own retrieval results.
    """
    lines = doc.page_content.splitlines()
    headings = [ln.lstrip("# ").strip() for ln in lines if ln.startswith("#")]

    opening = preview(doc.page_content, 46)
    if headings:
        return f'{opening}  [+ {" + ".join(headings)}]'
    return opening


# ---------------------------------------------------------------------------
# 4. DEMO
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading the embedding model (first run downloads ~50 MB)...")
    embeddings = FastEmbedAdapter()
    print("Ready.\n")

    # --- A. What does a vector actually look like? ------------------------
    print("=" * 72)
    print("A. WHAT AN EMBEDDING IS")
    print("=" * 72)
    vec = embeddings.embed_query("paid leave policy")
    print(f'  text:       "paid leave policy"')
    print(f"  dimensions: {len(vec)}")
    print(f"  first 8:    {[round(x, 4) for x in vec[:8]]}")
    print()
    print("  That's it. Every chunk becomes exactly this many numbers,")
    print("  whether it's 3 words or 900 characters.")

    # --- B. Prove that similar meaning = similar vector -------------------
    print()
    print("=" * 72)
    print("B. SIMILAR MEANING -> SIMILAR VECTOR")
    print("=" * 72)

    sentences = [
        "How many vacation days do I get?",   # 0
        "What is the annual paid leave allowance?",  # 1 - same meaning, no shared words
        "How do I reset my laptop password?",  # 2 - different topic
    ]
    vecs = embeddings.embed_documents(sentences)

    for i, s in enumerate(sentences):
        print(f"  [{i}] {s}")
    print()
    pairs = [(0, 1, "same meaning, almost no shared words"),
             (0, 2, "different topic"),
             (1, 2, "different topic")]
    for i, j, note in pairs:
        score = cosine_similarity(vecs[i], vecs[j])
        bar = "#" * int(score * 40)
        print(f"  [{i}] vs [{j}]  {score:.3f}  {bar:<40} {note}")

    print()
    print("  Sentences 0 and 1 share only the word 'do'/'I'-ish - yet they")
    print("  score high. THAT is meaning-based search. Keyword search would")
    print("  score them near zero.")

    # --- C. Build the store ----------------------------------------------
    print()
    print("=" * 72)
    print("C. BUILDING THE VECTOR STORE")
    print("=" * 72)
    store = build_vector_store()
    print(f"  Embedded and stored all chunks from docs/")

    # --- D. Search by meaning --------------------------------------------
    print()
    print("=" * 72)
    print("D. SEARCHING BY MEANING (not keyword)")
    print("=" * 72)

    queries = [
        ("How much vacation time do I get?",
         "docs never say 'vacation' - they say 'paid leave'"),
        ("Can I work from Spain for a month?",
         "docs never say 'Spain' - they say 'a country other than...'"),
        ("How precise is the robot arm?",
         "docs never say 'precise' - they say 'repeatability'"),
        ("What do I do if my laptop gets stolen?",
         "close vocabulary match, should be an easy win"),
    ]

    for query, note in queries:
        print(f"\n  QUERY: {query!r}")
        print(f"         ({note})")
        results = store.similarity_search_with_score(query, k=3)
        for rank, (doc, score) in enumerate(results, start=1):
            src = doc.metadata["source"]
            cid = doc.metadata["chunk_id"]
            print(f"    {rank}. {score:.3f}  [{src} #{cid}]  {describe(doc)}")

    # --- E. The failure case that motivates Step 6 ------------------------
    print()
    print("=" * 72)
    print("E. THE PROBLEM THAT STEP 6 WILL FIX")
    print("=" * 72)

    bad_query = "What is the wifi password in the office?"
    print(f"\n  QUERY: {bad_query!r}")
    print("         (the answer is NOWHERE in our documents)")
    results = store.similarity_search_with_score(bad_query, k=3)
    for rank, (doc, score) in enumerate(results, start=1):
        src = doc.metadata["source"]
        cid = doc.metadata["chunk_id"]
        print(f"    {rank}. {score:.3f}  [{src} #{cid}]  {describe(doc)}")

    print()
    print("  Look at that. The store returned 3 chunks anyway, with")
    print("  respectable-looking scores - because a vector store ALWAYS")
    print("  returns the k nearest chunks. It has no concept of 'nothing")
    print("  here is relevant'. Nearest is not the same as relevant.")
    print()
    print("  If we hand these to an LLM and say 'answer from this context',")
    print("  we invite a confident, wrong answer.")
    print()
    print("  That is exactly what the grade_docs node in Step 6 exists to")
    print("  catch. Remember this output when we get there.")
    print("=" * 72)
