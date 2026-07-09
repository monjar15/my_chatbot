import re                                                   # For stripping any stray <think> blocks from LLM output
from langchain_ollama import ChatOllama                    
from langchain_core.documents import Document               # LangChain Document (page_content + metadata)
from langchain_core.prompts import ChatPromptTemplate      # Template that formats messages for the LLM
from langchain_core.output_parsers import StrOutputParser  # Parses LLM ChatMessage output to a plain string
from typing import Generator                               # Type hint for generator functions (streaming)
from rag.logger import logger                              # Shared logger instance


# ── Configuration constants ──────────────────────────────────────────────────

LLM_MODEL   = "qwen3:1.7b"     # TIP: Use "qwen3:1.7b/no_think " to disable thinking mode for faster (but less reasoned) answers
LLM_TEMP    = 0              # Low temperature → more deterministic, factual answers (0 = fully deterministic)
LLM_TOKENS  = 4608           # Maximum number of tokens the LLM is allowed to generate per response
LLM_KEEP_ALIVE = "30m"       # Keep model loaded in VRAM between requests (reduces reload latency)
                             # Use -1 to keep alive indefinitely, or "0" to unload immediately after use
LLM_NUM_CTX = 16384          # Cap context window — measure your actual prompt size (context + history + question)
                             # and set this just above it; avoid leaving it unset (defaults can be larger than needed)
LLM_SEED = 42                # Fixed seed → combined with temperature=0, removes run-to-run sampling variance
LLM_REPEAT_PENALTY = 1.15    # Guards against repeated/looping list items on enumeration queries

# ── LLM initialiser ──────────────────────────────────────────────────────────

# Create one shared LLM instance for the entire application.
_SHARED_LLM = ChatOllama(
    model=LLM_MODEL,
    temperature=LLM_TEMP,
    num_predict=LLM_TOKENS,
    keep_alive=LLM_KEEP_ALIVE,
    num_ctx=LLM_NUM_CTX,
    seed=LLM_SEED,
    repeat_penalty=LLM_REPEAT_PENALTY,
)

def get_llm() -> ChatOllama:

    logger.info(                   # Record loaded LLM configuration
        f"LLM → model='{LLM_MODEL}', temp={LLM_TEMP}, max_tokens={LLM_TOKENS}, keep_alive={LLM_KEEP_ALIVE}"
    )

    print(f"[Chain] LLM → model='{LLM_MODEL}', temp={LLM_TEMP}, max_tokens={LLM_TOKENS}, keep_alive={LLM_KEEP_ALIVE}")  # Log LLM config

    return _SHARED_LLM                   


# ── Prompt template ───────────────────────────────────────────────────────────

def _build_prompt() -> ChatPromptTemplate:

    template = """\
You are a professional internal company assistant. \
Your sole knowledge source is the documents section below, \
which contains retrieved excerpts from official company documents.

──────────────────────────────────────────
DOCUMENTS:
{context}
──────────────────────────────────────────

CHAT HISTORY (most recent turns, for reference only):
{chat_history}

──────────────────────────────────────────

QUESTION: {question}

──────────────────────────────────────────
INSTRUCTIONS — follow in this exact order:

STEP 1 — SEARCH: CAREFULLY read EVERY sentence in the documents above \
and identify ALL passages that are RELEVANT to the QUESTION. \
You may use the CHAT HISTORY to understand follow-up questions \
(e.g. resolving pronouns like "it", "that", "they", "him", "her", "he", or "she"), \
but NEVER use history as a knowledge source.

STEP 2 — ANSWER: If relevant information is found, construct a clear, \
concise, and professional answer using ONLY those passages. \
Do not add any information that is not explicitly stated in the documents. \
Do not infer, assume, or speculate beyond what is written.

STEP 3 — FORMAT: Use bullet points only when listing multiple distinct items. \
Otherwise, respond in short, direct paragraphs. \
Answer as a knowledgeable assistant speaking in your own words — NEVER USE the \
words "chunks", "passages", "excerpts", "reference", or any other retrieval-related term \
when phrasing your answer. \
DO NOT say things like "According to the provided chunks 1, or 2 or 3, or 4", "According \
to the provided reference 1, or 2 or 3, or 4", or "Based on the passages above" — just \
state the information directly as fact.

STEP 4 — NOT FOUND: Only if the documents contains absolutely no information \
relevant to the QUESTION — after carefully completing STEP 1 — \
respond with exactly this message and nothing else (preserve the exact wording \
but fill in the bracketed placeholder):

"I wasn't able to find any results for **[restate the user's question here]**. \
Did you perhaps mistype your query? \
Please reply with **Yes** if you'd like to retype it, or **No** if the query was correct."

CRITICAL RULES:
- You must complete STEP 1 before concluding that information is absent.
- Never use knowledge from outside the documents.
- Never fabricate names, figures, policies, or procedures.
- Never say "I wasn't able to find..." if the documents contains a relevant \
  passage, even if the passage only partially addresses the question.
- Never add follow-up suggestions or escalation advice in this step — \
  that is handled separately if the user confirms the query is correct.

ANSWER:"""                                              # Multi-line f-string with placeholders for context and question

    return ChatPromptTemplate.from_template(template)  # Wrap string into a LangChain prompt object


# ── Context formatter ─────────────────────────────────────────────────────────

def _format_context(docs_with_scores: list[tuple[Document, float]]) -> str:

    parts: list[str] = []                              # Accumulate formatted chunk strings here

    for i, (doc, score) in enumerate(docs_with_scores, start=1):   # Enumerate from 1 
        source = doc.metadata.get(                     # Try to get the source file name from LlamaIndex metadata
            "file_name",                               # LlamaIndex usually stores the filename here
            doc.metadata.get("source", "unknown")      # Fall back to 'source' key or 'unknown' string
        )

        logger.info(                                   # Record retrieved source
            f"Chunk {i} | Score={score:+.4f} | Source={source}"
        )

        parts.append(                                  # Build a clearly delimited chunk block
            f"({source})\n"                            # Source
            f"{doc.page_content}"                      # The actual text content of the chunk
        )

    return "\n\n---\n\n".join(parts)                   # Join chunks with a visible separator for clarity


# ── Query rewriting (resolve pronouns/follow-ups before retrieval) ───────────

def rewrite_query_with_history(
    query: str,
    chat_history: list[dict] | None = None,
    max_history_turns: int = 3                           # Keep this short — only need enough context to resolve references
) -> str:

    pronouns = {
        "it", "its", "they", "them", "their",
        "this", "that", "these", "those",
        "he", "him", "his", "she", "her"
    }

    words = re.findall(r"\b\w+\b", query.lower())

    if not any(word in pronouns for word in words):
        return query

    llm = get_llm()                                      # Reuse the same LLM instance

    if chat_history is None:
        chat_history = []

    recent = chat_history[-(max_history_turns * 2):]

    history_lines = [
        f"{'User' if msg['role']=='user' else 'Assistant'}: {msg['content']}"
        for msg in recent
    ]

    history_str = "\n".join(history_lines)

    rewrite_prompt = f"""\
Given the CHAT HISTORY below, rewrite the FOLLOW-UP QUESTION into a fully \
standalone question that contains no pronouns or ambiguous references. \
Replace words like "he", "she", "it", "they", "that", "this" with the \
specific name or subject they refer to.

IMPORTANT: Pronouns almost always refer to the subject of the MOST RECENT \
exchange (the last User/Assistant pair) in the CHAT HISTORY below — NOT the \
subject of earlier exchanges. Only fall back to an earlier subject if the \
most recent exchange clearly does not introduce a person/subject that fits \
the pronoun.

If the FOLLOW-UP QUESTION is already standalone, return it unchanged. \
Respond with ONLY the rewritten question and nothing else — no explanation, \
no quotation marks, no preamble.

CHAT HISTORY (oldest → newest; the LAST exchange is the most relevant one):
{history_str}

MOST RECENT SUBJECT HINT: the last User/Assistant exchange above is what \
"he/she/it/they/that/this" in the FOLLOW-UP QUESTION most likely refers to.

FOLLOW-UP QUESTION: {query}

REWRITTEN QUESTION:"""

    raw_output = str(llm.invoke(rewrite_prompt).content).strip()  # Single blocking call

    # ── Defensive cleanup: strip any <think>...</think> block in case thinking mode still fired ──
    if "<think>" in raw_output:
        raw_output = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()
        logger.warning(
            f"Rewrite output contained a <think> block — stripped before use. Raw length was {len(raw_output)} chars after cleanup"
        )

    logger.info(f"Query rewrite: '{query}' → '{raw_output}'")

    print(f"[Chain] Query rewrite: '{query}' → '{raw_output}'")  # Log rewrite for debugging

    return raw_output or query                                  # Fallback to original if rewrite came back empty


# ── Streaming RAG call ────────────────────────────────────────────────────────

def run_rag_chain_stream(
    query: str,
    docs_with_scores: list[tuple[Document, float]],
    chat_history: list[dict] | None = None,            # list of {"role": .., "content": ..} dicts
    max_history_turns: int = 4                         # how many past exchanges to include
) -> Generator[str, None, None]:

    llm     = get_llm()                                # Initialise the LLM
    prompt  = _build_prompt()                          # Get the prompt template

    MIN_SCORE = 0.30
    MAX_CONTEXT_DOCS = 8

    filtered_docs = [
        (doc, score)
        for doc, score in docs_with_scores
        if score >= MIN_SCORE
    ][:MAX_CONTEXT_DOCS]

    if not filtered_docs:
        filtered_docs = docs_with_scores[:MAX_CONTEXT_DOCS]

    context = _format_context(filtered_docs or docs_with_scores)        # Format retrieved chunks into a context string

    # ── Format chat history into a readable string for the prompt ──
    # Each turn is labelled User/Assistant and separated by a blank line.
    # We trim to the last `max_history_turns` exchanges (1 exchange = 1 user + 1 assistant msg).
    if chat_history:
        recent   = chat_history[-(max_history_turns * 2):]          # Slice to keep N most recent turns
        history_lines = []
        for msg in recent:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            history_lines.append(f"{role_label}: {msg['content']}")
        history_str = "\n".join(history_lines)                       # One line per message
    else:
        history_str = "(No prior conversation)"                      # Placeholder when history is empty

    # ── Build LCEL chain (same structure as non-streaming) ──
    chain = prompt | llm | StrOutputParser()           # Pipe: prompt template → LLM → string parser
    
    logger.info(                                       # Record stream start
        f"Streaming RAG chain for: '{query}' | history_turns={len(chat_history or [])}"
    )

    print(f"[Chain] Streaming RAG chain for: '{query}' | history_turns={len(chat_history or [])}")                                                 # Log stream start

    token_count = 0                                    # Counter to track total streamed tokens

    # ── Thinking-mode filter state ──
    # Qwen3 emits <think>...</think> before its real answer when thinking mode
    # is on. We must not let raw reasoning leak onto the user's screen, but the
    # tags can arrive split across separate stream chunks (e.g. "<th" + "ink>"),
    # so we buffer and only release text once we're sure it isn't part of a tag.
    OPEN_TAG, CLOSE_TAG = "<think>", "</think>"
    in_think = False                                    # True while inside a <think>...</think> block
    buffer   = ""                                       # Holds text not yet safe to release

    def _partial_suffix_len(s: str, tag: str) -> int:
        # Longest suffix of s that could be the start of `tag` (so we can hold it back)
        for n in range(min(len(s), len(tag) - 1), 0, -1):
            if s.endswith(tag[:n]):
                return n
        return 0

    for token in chain.stream({                        # .stream() yields partial text chunks instead of blocking
        "context":  context,                           # Formatted retrieved passages
        "question": query,                             # User's question
        "chat_history": history_str,                   # pass formatted history to prompt
    }):
        token_count += 1                               # Increment token counter
        buffer += token                                # ← feed the token into the filter buffer caller

        while True:
            if not in_think:
                if OPEN_TAG in buffer:                                   # Full opening tag found
                    pre, buffer = buffer.split(OPEN_TAG, 1)
                    if pre:
                        yield pre                                        # Release text before the tag
                    in_think = True
                    continue
                hold = _partial_suffix_len(buffer, OPEN_TAG)             # Might be a partial "<think>" at the tail
                safe = buffer[: len(buffer) - hold] if hold else buffer
                if safe:
                    yield safe                                           # Safe to show
                    buffer = buffer[len(safe):]
                break
            else:
                if CLOSE_TAG in buffer:                                  # Full closing tag found
                    _, buffer = buffer.split(CLOSE_TAG, 1)
                    in_think = False
                    continue
                hold = _partial_suffix_len(buffer, CLOSE_TAG)            # Might be a partial "</think>" at the tail
                buffer = buffer[len(buffer) - hold:] if hold else ""     # Discard reasoning text, keep only the hold-back
                break

    if buffer and not in_think:                        # Flush any trailing safe text after the stream ends
        yield buffer
    elif buffer and in_think:                           # Model never closed its <think> block (e.g. hit LLM_TOKENS cap)
        logger.warning("Stream ended while still inside a <think> block — reasoning was truncated and discarded")

    logger.info(                                       # Record stream completion
        f"Stream complete ({token_count} token(s))"
    )
        
    print(f"[Chain] ✅ Stream complete ({token_count} tokens)")         # Log when streaming finishes