"""
Step 3 - Load & chunk documents
===============================
Read real files from docs/ and split them into retrieval-sized pieces.

Still no LLM and no API key. This is pure text processing.

Run it:   python step3_load_and_chunk.py
"""

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_DIR = Path(__file__).parent / "docs"

# --- The two numbers that matter most in all of RAG -------------------------
# Measured in CHARACTERS (not tokens). Rough guide: 1 token ~= 4 characters,
# so 1000 characters is roughly 250 tokens.
CHUNK_SIZE = 1000     # about a paragraph or two: big enough to stand alone,
                      # small enough to be about ONE idea
CHUNK_OVERLAP = 200   # 20% - each chunk repeats the tail of the previous one
                      # so an idea split across a boundary isn't lost


# ---------------------------------------------------------------------------
# 1. LOADING - turn files on disk into Document objects
# ---------------------------------------------------------------------------
def load_documents(docs_dir: Path = DOCS_DIR) -> list[Document]:
    """Read every .md, .txt and .pdf in docs/ into a list of Documents.

    A Document is just two fields:
        page_content -> the text (this is what gets embedded and searched)
        metadata     -> a free-form dict that rides along untouched

    We put the filename in metadata NOW so that at answer time, five steps
    later, we can still say "according to handbook.md". Metadata is how you
    get citations.
    """
    documents: list[Document] = []

    for path in sorted(docs_dir.iterdir()):
        suffix = path.suffix.lower()

        if suffix in {".md", ".txt"}:
            text = path.read_text(encoding="utf-8")
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": path.name},
                )
            )

        elif suffix == ".pdf":
            # Imported here so you don't need pypdf installed unless you
            # actually have PDFs.
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            # One Document per PAGE, so we can cite a page number later.
            # See cite() in prompts.py - "it_policy.pdf p.1 #6" beats pointing
            # at a 200-page file.
            pages_with_text = 0
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if not text.strip():
                    continue  # blank, or an image-only page - see warning below
                pages_with_text += 1
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": path.name, "page": page_number},
                    )
                )

            # A SCANNED PDF is a picture of text, not text. pypdf extracts
            # nothing from it, and the failure is silent: the file loads, no
            # error is raised, and the assistant simply never knows anything
            # about it. Warn loudly, because otherwise you will spend an hour
            # tuning chunk sizes for a document that was never ingested.
            if pages_with_text == 0:
                print(f"  WARNING: {path.name} has {len(reader.pages)} page(s) "
                      f"but no extractable text.")
                print(f"           It is probably a SCANNED pdf (an image). "
                      f"You need OCR - e.g. ocrmypdf - to use it.")
            elif pages_with_text < len(reader.pages):
                print(f"  note: {path.name} - {len(reader.pages) - pages_with_text} "
                      f"of {len(reader.pages)} page(s) had no text (blank or scanned)")

        # anything else (.gitkeep, images, ...) is ignored

    return documents


# ---------------------------------------------------------------------------
# 2. CHUNKING - split big Documents into retrieval-sized pieces
# ---------------------------------------------------------------------------
def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split each Document into ~CHUNK_SIZE character pieces.

    RecursiveCharacterTextSplitter tries a priority list of separators and
    only falls back to the next one when a piece is still too big:

        "\\n\\n"  paragraph breaks   <- try hardest to split here
        "\\n"    line breaks
        " "     spaces (word boundaries)
        ""      give up, cut mid-word

    That's why chunks land on natural boundaries instead of starting
    mid-senten-
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # length_function=len is the default: counts characters.
        # Swap in a real token counter later if you're optimising cost.
        add_start_index=True,  # records where in the original the chunk began
    )

    # split_documents() carries the metadata from each parent Document over
    # to all of its children automatically. That's why "source" survives.
    chunks = splitter.split_documents(documents)

    # Add our own chunk number - handy for debugging and for citations.
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks


# ---------------------------------------------------------------------------
# 3. DEMO
# ---------------------------------------------------------------------------
def _preview(text: str, width: int = 68) -> str:
    """One-line preview of a chunk, with newlines made visible."""
    flat = text.replace("\n", " ").strip()
    return flat[:width] + ("..." if len(flat) > width else "")


if __name__ == "__main__":
    print("=" * 72)
    print("LOADING")
    print("=" * 72)

    docs = load_documents()
    if not docs:
        print("  No documents found in docs/. Add some .md, .txt or .pdf files.")
        raise SystemExit(1)

    for d in docs:
        page = f" p.{d.metadata['page']}" if "page" in d.metadata else ""
        print(f"  {d.metadata['source'] + page:22} {len(d.page_content):>6,} characters")
    total = sum(len(d.page_content) for d in docs)
    print(f"  {'TOTAL':20} {total:>6,} characters across {len(docs)} document(s)")

    print()
    print("=" * 72)
    print(f"CHUNKING   (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print("=" * 72)

    chunks = chunk_documents(docs)

    sizes = [len(c.page_content) for c in chunks]
    print(f"  {len(chunks)} chunks created")
    print(f"  smallest: {min(sizes):,} chars")
    print(f"  largest:  {max(sizes):,} chars")
    print(f"  average:  {sum(sizes) // len(sizes):,} chars")

    print()
    print("  Note the largest chunk is <= CHUNK_SIZE, but the average is")
    print("  lower - the splitter stops early at a paragraph break rather")
    print("  than padding to exactly 1000 characters.")

    print()
    print("=" * 72)
    print("THE FIRST 5 CHUNKS")
    print("=" * 72)
    for c in chunks[:5]:
        m = c.metadata
        print(f"\n  [chunk {m['chunk_id']}]  {m['source']}  "
              f"(starts at char {m['start_index']}, {len(c.page_content)} chars)")
        print(f"    {_preview(c.page_content)}")

    print()
    print("=" * 72)
    print("PROOF THAT OVERLAP WORKS")
    print("=" * 72)
    print("  The END of chunk 0 should reappear at the START of chunk 1.\n")
    a, b = chunks[0].page_content, chunks[1].page_content
    print(f"  chunk 0 ends with ...: {_preview(a[-CHUNK_OVERLAP:])}")
    print(f"  chunk 1 starts with .: {_preview(b[:CHUNK_OVERLAP])}")

    # Find the actual longest shared boundary text.
    shared = ""
    for n in range(min(len(a), len(b)), 0, -1):
        if a[-n:] == b[:n]:
            shared = a[-n:]
            break
    print(f"\n  Literally repeated: {len(shared)} characters")
    print(f"    -> {_preview(shared)}")

    print()
    print("=" * 72)
    print("METADATA ON EVERY CHUNK (this is what makes citations possible)")
    print("=" * 72)
    print(f"  {chunks[0].metadata}")
    print()
    print("  Each chunk still knows which file it came from, even though the")
    print("  original Document was torn into pieces. split_documents() copies")
    print("  the parent's metadata onto every child automatically.")
    print("=" * 72)
