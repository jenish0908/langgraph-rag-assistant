"""
The language model - configured in ONE place.

Every node imports get_llm() from here, so switching provider or model is a
one-file change. Nothing in the graph knows or cares which provider is in use.
That is the same lesson as the embeddings adapter in Step 4: program against
the interface, not the vendor.

WHICH PROVIDER RUNS
-------------------
Auto-detected from whichever API key is present, so local development stays
offline and a deployment "just works" once you set a secret:

    GROQ_API_KEY        -> Groq        (free, very fast)
    GOOGLE_API_KEY      -> Gemini      (free, generous limits)
    ANTHROPIC_API_KEY   -> Claude      (paid, best quality)
    (none of the above) -> Ollama      (local, offline, free, slow on CPU)

Force one explicitly with LLM_PROVIDER=groq|gemini|anthropic|ollama.
"""

import os

# ---------------------------------------------------------------------------
# MODEL PER PROVIDER
# ---------------------------------------------------------------------------
MODELS = {
    # Local. Benchmarked on this machine:
    #   gemma3:1b  - fast, but FAILED the relevance-grading task. Unusable.
    #   gemma3:4b  - correct on every test. ~6 tok/s on CPU. This one.
    #   gemma4     - 9.6 GB vs 7.8 GB free RAM -> swaps to disk. Too slow.
    "ollama": "gemma3:4b",

    # Groq: free tier, no credit card, ~280 tok/s on this model.
    # Alternatives: "openai/gpt-oss-120b", "llama-3.1-8b-instant" (faster/weaker)
    "groq": "llama-3.3-70b-versatile",

    # Google Gemini: free tier, generous daily limits.
    # Alternatives: "gemini-3.5-flash-lite" (cheaper/faster), "gemini-3.8-flash"
    "gemini": "gemini-2.5-flash",

    # Anthropic: paid, no free tier. Best grading quality of the four.
    # Alternatives: "claude-sonnet-5" ($2/$10 per Mtok),
    #               "claude-haiku-4-5" ($1/$5 per Mtok)
    "anthropic": "claude-opus-5",   # $5/$25 per Mtok
}


def detect_provider() -> str:
    """Which provider to use, from env vars. Explicit setting wins."""
    forced = os.getenv("LLM_PROVIDER", "").strip().lower()
    if forced:
        if forced not in MODELS:
            raise ValueError(
                f"LLM_PROVIDER={forced!r} is not one of {sorted(MODELS)}")
        return forced

    if os.getenv("GROQ_API_KEY"):
        return "groq"
    if os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "ollama"


def describe() -> str:
    """Human-readable 'provider / model', for the UI and the logs."""
    p = detect_provider()
    return f"{p} / {MODELS[p]}"


def get_llm(max_tokens: int = 300, temperature: float = 0.0):
    """Build the chat model for the active provider.

    temperature=0
        Randomness. Creative writing wants some; "what does this document say"
        wants the most probable answer, and the same answer twice. Always 0
        for factual extraction.

    max_tokens
        A cap on output length. It matters enormously on Ollama (CPU
        generation is ~6 tok/s, so an unbounded model that decides to write
        six paragraphs costs you a minute) and barely at all on a hosted API.

        On hosted providers we therefore RAISE it to a floor of 1024. A cap
        that is too tight truncates an answer mid-sentence, and you only pay
        for tokens actually produced - so a generous cap is nearly free.
        Tight caps are a local-hardware workaround, not a virtue.
    """
    provider = detect_provider()
    model = MODELS[provider]

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
            temperature=temperature,
            num_predict=max_tokens,
            keep_alive="10m",
            # Measured on this machine (4 physical cores / 8 threads). Ollama
            # defaults to physical cores; all 8 logical threads is clearly
            # faster, A/B'd over 6 novel questions:
            #   num_thread=4:  prefill 15.8 tok/s  generate 4.2 tok/s  61s avg
            #   num_thread=8:  prefill 26.6 tok/s  generate 5.9 tok/s  40s avg
            # Re-measure on other hardware; more threads is not automatically
            # better.
            num_thread=8,
        )

    cloud_tokens = max(max_tokens, 1024)

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, temperature=temperature,
                        max_tokens=cloud_tokens)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, temperature=temperature,
                                      max_output_tokens=cloud_tokens)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, temperature=temperature,
                             max_tokens=cloud_tokens)

    raise ValueError(f"unknown provider {provider!r}")
