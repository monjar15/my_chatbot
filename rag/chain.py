from langchain_ollama import ChatOllama                    
from langchain_core.documents import Document                      # LangChain Document (page_content + metadata)
from langchain_core.prompts import ChatPromptTemplate      # Template that formats messages for the LLM
from langchain_core.output_parsers import StrOutputParser  # Parses LLM ChatMessage output to a plain string
from typing import Generator                               # Type hint for generator functions (streaming)
from rag.logger import logger                              # Shared logger instance


# ── Configuration constants ──────────────────────────────────────────────────

LLM_MODEL   = "qwen3:1.7b"   # Ollama model tag – must be pulled: `ollama pull qwen3:8b`
                             # TIP: Use "qwen3:1.7b/no_think " to disable thinking mode for faster (but less reasoned) answers
LLM_TEMP    = 0              # Low temperature → more deterministic, factual answers (0 = fully deterministic)
LLM_TOKENS  = 1024           # Maximum number of tokens the LLM is allowed to generate per response
LLM_KEEP_ALIVE = "10m"       # Keep model loaded in VRAM between requests (reduces reload latency)
                             # Use -1 to keep alive indefinitely, or "0" to unload immediately after use

# ── LLM initialiser ──────────────────────────────────────────────────────────

def get_llm() -> ChatOllama:

    llm = ChatOllama(              
        model=LLM_MODEL,           
        temperature=LLM_TEMP,      
        num_predict=LLM_TOKENS,
        keep_alive=LLM_KEEP_ALIVE,    
    )

    logger.info(                   # Record loaded LLM configuration
        f"LLM → model='{LLM_MODEL}', temp={LLM_TEMP}, max_tokens={LLM_TOKENS}, keep_alive={LLM_KEEP_ALIVE}"
    )

    print(f"[Chain] LLM → model='{LLM_MODEL}', temp={LLM_TEMP}, max_tokens={LLM_TOKENS}, keep_alive={LLM_KEEP_ALIVE}")  # Log LLM config

    return llm                   


# ── Prompt template ───────────────────────────────────────────────────────────

def _build_prompt() -> ChatPromptTemplate:

    template = """\
You are a professional internal company assistant. \
Your sole knowledge source is the CONTEXT section below, \
which contains retrieved excerpts from official company documents.

──────────────────────────────────────────
CONTEXT:
{context}
──────────────────────────────────────────

CHAT HISTORY (most recent turns, for reference only):
{chat_history}

──────────────────────────────────────────

QUESTION: {question}

──────────────────────────────────────────
INSTRUCTIONS — follow in this exact order:

STEP 1 — SEARCH: Carefully read every sentence in the CONTEXT above \
and identify all passages that are relevant to the QUESTION. \
You may use the CHAT HISTORY to understand follow-up questions \
(e.g. resolving pronouns like "it", "that", "they", "him", "her", "he", or "she"), \
but NEVER use history as a knowledge source.

STEP 2 — ANSWER: If relevant information is found, construct a clear, \
concise, and professional answer using ONLY those passages. \
Do not add any information that is not explicitly stated in the CONTEXT. \
Do not infer, assume, or speculate beyond what is written.

STEP 3 — FORMAT: Use bullet points only when listing multiple distinct items. \
Otherwise, respond in short, direct paragraphs.

STEP 4 — NOT FOUND: Only if the CONTEXT contains absolutely no information \
relevant to the QUESTION — after carefully completing STEP 1 — \
respond with exactly this message and nothing else (preserve the exact wording \
but fill in the bracketed placeholder):

"I wasn't able to find any results for **[restate the user's question here]**. \
Did you perhaps mistype your query? \
Please reply with **Yes** if you'd like to retype it, or **No** if the query was correct."

CRITICAL RULES:
- You must complete STEP 1 before concluding that information is absent.
- Never use knowledge from outside the CONTEXT.
- Never fabricate names, figures, policies, or procedures.
- Never say "I wasn't able to find..." if the CONTEXT contains a relevant \
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
            f"[Chunk {i} | Relevance Score: {score:+.4f} | Source: {source}]\n"  # Header with rank, score, source
            f"{doc.page_content}"                      # The actual text content of the chunk
        )

    return "\n\n---\n\n".join(parts)                   # Join chunks with a visible separator for clarity


# ── Streaming RAG call ────────────────────────────────────────────────────────

def run_rag_chain_stream(
    query: str,
    docs_with_scores: list[tuple[Document, float]],
    chat_history: list[dict] | None = None,            # list of {"role": .., "content": ..} dicts
    max_history_turns: int = 5                         # how many past exchanges to include
) -> Generator[str, None, None]:

    llm     = get_llm()                                # Initialise the LLM
    prompt  = _build_prompt()                          # Get the prompt template
    context = _format_context(docs_with_scores)        # Format retrieved chunks into a context string

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

    for token in chain.stream({                        # .stream() yields partial text chunks instead of blocking
        "context":  context,                           # Formatted retrieved passages
        "question": query,                              # User's question
        "chat_history": history_str,                   # pass formatted history to prompt
    }):
        token_count += 1                               # Increment token counter
        yield token                                    # Yield the current token to the Streamlit caller

    logger.info(                                       # Record stream completion
        f"Stream complete ({token_count} token(s))"
    )
        
    print(f"[Chain] ✅ Stream complete ({token_count} tokens)")         # Log when streaming finishes