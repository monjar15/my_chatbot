from langchain_community.retrievers import BM25Retriever   
from langchain_core.documents import Document               # LangChain's Document wrapper (page_content + metadata)
from langchain_qdrant import QdrantVectorStore              
from sentence_transformers import CrossEncoder              
from rag.logger import logger                               # Import shared logger
import streamlit as st
import re                                                   # For enumeration-query pattern detection

# ── Configuration constants ──────────────────────────────────────────────────

RERANKER_MODEL  = "BAAI/bge-reranker-v2-m3"
TOP_K_RETRIEVE_NARROW = 10                      # Default candidate pool for single-fact queries — fast path
TOP_N_RERANK_NARROW   = 4                       # Default rerank count for single-fact queries — fast path
TOP_K_RETRIEVE_WIDE   = 20                      # Widened candidate pool for enumeration-style queries (e.g.
                                                # "who are all the X") so every distinct entity has a chance
                                                # to enter reranking                            
TOP_N_RERANK_WIDE     = 15                      # Widened rerank count so multi-entity answers aren't cut off

# ── Patterns that signal a "list everything" style question rather than ──
# ── a single-fact lookup — only these get the slower, wider retrieval path ──
_ENUMERATION_PATTERNS = re.compile(
    r"\b(who are|what are|list|all the|every|each of|name the|which (ladies|people|persons|women|men|items))\b",
    re.IGNORECASE
)


def _is_enumeration_query(query: str) -> bool:
    return bool(_ENUMERATION_PATTERNS.search(query))


# ── Module-level cache for cross-encoder ────────────────────────────────────

_cross_encoder_cache: CrossEncoder | None = None        # Will hold the loaded cross-encoder after first call


def _get_cross_encoder() -> CrossEncoder:

    global _cross_encoder_cache                         # Allow writing to the module-level variable

    if _cross_encoder_cache is None:                    # Only load if not already cached
        print(f"[Retrieval] Loading cross-encoder '{RERANKER_MODEL}' ...")  # First-time load notification
        _cross_encoder_cache = CrossEncoder(RERANKER_MODEL)                 # Download/load the cross-encoder model
        print(f"[Retrieval] ✅ Cross-encoder loaded.")                      # Confirm load

    return _cross_encoder_cache                                             # Return the (possibly cached) model


# ── Stage 1: BM25 retriever ──────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_bm25_retriever(chunks: list[Document]) -> BM25Retriever:

    retriever = BM25Retriever.from_documents(chunks)   # Build the BM25 index by tokenising every chunk's page_content
    retriever.k = TOP_K_RETRIEVE_NARROW                       

    print(f"[Retrieval] BM25 index built from {len(chunks)} chunks (top-k={TOP_K_RETRIEVE_NARROW})")  # Log setup

    return retriever                                   


# ── Stage 2: Dense retriever ─────────────────────────────────────────────────

def get_dense_retriever(vectorstore: QdrantVectorStore, k: int = TOP_K_RETRIEVE_NARROW):

    retriever = vectorstore.as_retriever(              
        search_type="similarity",                      
        search_kwargs={"k": k}                         
    )

    print(f"[Retrieval] Dense retriever configured (top-k={k})")  # Log setup

    return retriever                                   


# ── Merge helper ─────────────────────────────────────────────────────────────

def _merge_results(bm25_docs: list[Document], dense_docs: list[Document]) -> list[Document]:

    # Calculate how many docs to take from each source based on the desired ratio
    # BM25 gets 20% of the total pool, dense gets 80%
    total = len(bm25_docs) + len(dense_docs)          # Total number of retrieved docs across both sources

    bm25_limit = round(total * 0.20)                  # 20% allocated to BM25 results
    dense_limit = round(total * 0.80)                 # 80% allocated to dense (vector) results

    # Slice each list to respect the ratio limits (preserve original ranking order within each source)
    bm25_slice  = bm25_docs[:bm25_limit]              # Take only the top 20% from BM25
    dense_slice = dense_docs[:dense_limit]            # Take only the top 80% from dense

    # Concatenate dense first so semantic results lead, followed by BM25 keyword results
    combined: list[Document] = dense_slice + bm25_slice   # Dense first to prioritize semantic results

    # ── Deduplicate by page_content so the same chunk from both sources ──
    # ── doesn't consume two slots in the final reranked top_n ──
    seen: set[str] = set()
    merged: list[Document] = []
    for doc in combined:
        key = doc.page_content.strip()
        if key not in seen:
            seen.add(key)
            merged.append(doc)

    return merged                                     # Return the weighted merged list


# ── Stage 3: Cross-encoder re-ranking ────────────────────────────────────────

def rerank_documents(
    query: str,
    docs: list[Document],
    top_n: int = TOP_K_RETRIEVE_NARROW
) -> list[tuple[Document, float]]:

    cross_encoder = _get_cross_encoder()               # Load (or retrieve cached) cross-encoder model

    # Build input pairs: each pair is (query, document_text)
    pairs = [                                          # List comprehension to create input pairs
        (query, doc.page_content)                      # Cross-encoder expects [query, passage] pairs
        for doc in docs                                # One pair per candidate document
    ]

    scores: list[float] = cross_encoder.predict(pairs).tolist()  # Score every (query, doc) pair in one batch call

    # Zip documents with their scores and sort best-first
    ranked = sorted(                                   # Sort the combined list
        zip(docs, scores),                             # ... of (Document, score) tuples ...
        key=lambda pair: pair[1],                      # ... by the score (second element of each tuple) ...
        reverse=True                                   # ... in descending order (highest score = most relevant)
    )

    top_docs = list(ranked)[:top_n]                    # Keep only the top-n documents after sorting

    print(f"[Retrieval] ✅ Returning top-{top_n} reranked document(s)")  # Log final selection

    return top_docs                                    # Return list of (Document, score) tuples


# ── Public pipeline entry point ───────────────────────────────────────────────

def retrieve(
    query: str,
    bm25_retriever: BM25Retriever,
    dense_retriever
) -> list[tuple[Document, float]]:
    
    logger.info(  # Record incoming user query
        f"Query received: {query}"
    )

    # ── Decide retrieval width based on query type ──────────────────────────
    is_enum = _is_enumeration_query(query)
    top_k   = TOP_K_RETRIEVE_WIDE if is_enum else TOP_K_RETRIEVE_NARROW
    top_n   = TOP_N_RERANK_WIDE   if is_enum else TOP_N_RERANK_NARROW

    logger.info(
        f"Query type: {'enumeration (wide)' if is_enum else 'narrow'} | top_k={top_k}, top_n={top_n}"
    )

    print(f"[Retrieval] Query: '{query}'")                              # Echo the query for debugging

    # ── Stage 1 & 2: Run both retrievers ─────────────────────────────────────
    bm25_retriever.k = top_k                                            # Adjust BM25 k per query type
    bm25_docs  = bm25_retriever.invoke(query)                           # BM25 keyword search

    logger.info(                                                        # Record BM25 retrieval count
        f"BM25 returned {len(bm25_docs)} documents"
    )

    dense_docs = dense_retriever.invoke(query, k=top_k)                 # Dense vector search

    logger.info(                                                        # Record dense retrieval count
        f"Dense returned {len(dense_docs)} documents"
    )

    print(f"[Retrieval] BM25 returned  {len(bm25_docs)} doc(s)")        # Log BM25 hit count
    print(f"[Retrieval] Dense returned {len(dense_docs)} doc(s)")       # Log dense hit count

    # ── Stage 3: Merge and deduplicate ───────────────────────────────────────
    merged = _merge_results(bm25_docs, dense_docs)                      # Combine and deduplicate results

    logger.info(                                                        # Record merged retrieval count
        f"Merged into {len(merged)} unique documents"
    )

    print(f"[Retrieval] After merge: {len(merged)} unique doc(s)")      # Log merged count

    # ── Stage 4: Cross-encoder re-ranking ────────────────────────────────────
    ranked = rerank_documents(query, merged, top_n)                     # Score and sort all candidates

    logger.info(                                                        # Record reranking results
        f"Returned {len(ranked)} reranked document(s)"                  # Final reranked document count
    )

    return ranked                                                        # Return final (Document, score) pairs
