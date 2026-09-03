"""
Shared prompts and formatting helpers.

WHY THIS FILE EXISTS - a bug worth understanding:

Step 6 originally did this:

    from step5_first_rag import PROMPT, format_docs

which looks harmless. But importing a module RUNS ITS TOP-LEVEL CODE, and
step5_first_rag.py has this at module level:

    STORE = build_vector_store()
    LLM   = get_llm(max_tokens=250)

So step 6 silently built the vector store TWICE and created two model objects.
The giveaway was "Building the vector store (once)..." printing twice.

Nothing crashed. The answers were correct. It was just quietly doing double
work - which is exactly the kind of bug that survives to production.

THE RULE: a module that others import should be safe to import. Put expensive
setup behind `if __name__ == "__main__":`, or behind a function, or in a
module like this one that holds only cheap, shared definitions.
"""

import re

from langchain_core.documents import Document

ANSWER_PROMPT = """You are a helpful assistant answering questions about internal company documents.

Use ONLY the context below to answer the question. Do not use any other knowledge.

If the context does not contain the answer, reply with exactly:
I don't know based on the provided documents.

Cite the source for each fact in square brackets exactly as it appears
above the context, for example [handbook.md #3] or [it_policy.pdf p.1 #11].
Keep the answer to 3 sentences or fewer.

Context:
{context}

Question: {question}

Answer:"""


def cite(doc: Document) -> str:
    """The label a chunk is cited by.

    PDFs are loaded one Document PER PAGE, so they carry a `page` in metadata.
    Using it here is the entire reason for loading them that way - otherwise a
    citation points at a 200-page file and helps nobody.

        handbook.md #3            (markdown - no pages)
        it_policy.pdf p.1 #11     (PDF - page 1)
    """
    m = doc.metadata
    page = f" p.{m['page']}" if "page" in m else ""
    return f"{m['source']}{page} #{m['chunk_id']}"


def format_docs(docs: list[Document]) -> str:
    """Turn retrieved chunks into labelled context text.

    The model can only cite what it can SEE, so we stamp each chunk with its
    source. This is where the metadata attached back in Step 3 earns its
    place: disk -> chunking -> vector store -> prompt -> citation.
    """
    blocks = []
    for d in docs:
        blocks.append(f"[{cite(d)}]\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)


# ---------------------------------------------------------------------------
# GRADING (used by Step 6 and Step 7)
# ---------------------------------------------------------------------------

# How much of each chunk the grader sees.
#
# This was 450 as a speed optimisation - and it was a BUG. The home-office
# stipend sits at character 724 of chunk 2, so the grader never saw it and
# rejected a chunk that DID contain the answer. Measured:
#
#   450  chars: 0/3 kept on 3 answerable questions   18s per grading call
#   1200 chars: correct on all 3                     28s per grading call
#
# 1.6x slower, and right instead of wrong. Correctness first; optimise only
# what you have measured. (1200 > our largest chunk of 975, so nothing is cut.)
GRADE_DOC_CHARS = 1200

GRADE_PROMPT = """You are a strict relevance grader.

For each numbered document below, decide whether it contains information that
actually helps answer the question. Be strict: being about the same general
topic is NOT enough. The document must contain the specific fact asked for.

{context}

Question: {question}

Reply with ONLY one line, one verdict per document, in this exact format:
1:yes 2:no 3:yes

Answer:"""


def number_docs(docs: list[Document]) -> str:
    """Compact, numbered rendering of the chunks for the grader."""
    return "\n\n".join(
        f"Document {i}:\n{d.page_content[:GRADE_DOC_CHARS]}"
        for i, d in enumerate(docs, start=1)
    )


def parse_verdicts(text: str, expected: int) -> list[bool]:
    """Turn '1:yes 2:no 3:yes' into [True, False, True].

    WHY PARSE TEXT INSTEAD OF USING with_structured_output()?

    We tried it. On gemma3:4b it returned an EMPTY list every time, silently.
    An empty list means "nothing is relevant", so the bug would have looked
    like a working grader that rejects everything - the worst kind of failure,
    because it is plausible.

    Small local models are much better at "reply with one line in this format"
    than at filling a JSON schema. On a big cloud model, prefer
    with_structured_output(). Test, don't assume.
    """
    found = dict(re.findall(r"(\d+)\s*:\s*(yes|no)", text.lower()))
    # Default to False (drop the doc) when the model didn't mention it.
    return [found.get(str(i), "no") == "yes" for i in range(1, expected + 1)]
