# rag/retrieval.py

from langchain_community.retrievers import BM25Retriever   
from langchain_core.documents import Document               # LangChain's Document wrapper (page_content + metadata)
from langchain_qdrant import QdrantVectorStore              
from sentence_transformers import CrossEncoder              


# ── Configuration constants ──────────────────────────────────────────────────

RERANKER_MODEL  = "BAAI/bge-reranker-v2-m3"  
TOP_K_RETRIEVE  = 10                           
TOP_N_RERANK    = 4                            


# ── Module-level cache for cross-encoder ────────────────────────────────────

_cross_encoder_cache: CrossEncoder | None = None  # Will hold the loaded cross-encoder after first call


def _get_cross_encoder() -> CrossEncoder:

    global _cross_encoder_cache                         # Allow writing to the module-level variable

    if _cross_encoder_cache is None:                    # Only load if not already cached
        print(f"[Retrieval] Loading cross-encoder '{RERANKER_MODEL}' ...")  # First-time load notification
        _cross_encoder_cache = CrossEncoder(RERANKER_MODEL)   # Download/load the cross-encoder model
        print(f"[Retrieval] ✅ Cross-encoder loaded.")          # Confirm load

    return _cross_encoder_cache                         # Return the (possibly cached) model


# ── Stage 1: BM25 retriever ──────────────────────────────────────────────────

def get_bm25_retriever(chunks: list[Document]) -> BM25Retriever:

    retriever = BM25Retriever.from_documents(chunks)   # Build the BM25 index by tokenising every chunk's page_content
    retriever.k = TOP_K_RETRIEVE                       

    print(f"[Retrieval] BM25 index built from {len(chunks)} chunks (top-k={TOP_K_RETRIEVE})")  # Log setup

    return retriever                                   


# ── Stage 2: Dense retriever ─────────────────────────────────────────────────

def get_dense_retriever(vectorstore: QdrantVectorStore, k: int = TOP_K_RETRIEVE):

    retriever = vectorstore.as_retriever(              
        search_type="similarity",                      
        search_kwargs={"k": k}                         
    )

    print(f"[Retrieval] Dense retriever configured (top-k={k})")  # Log setup

    return retriever                                   


# ── Merge helper ─────────────────────────────────────────────────────────────

def _merge_results(bm25_docs: list[Document], dense_docs: list[Document]) -> list[Document]:

    seen: set[str] = set()                             # Set of already-seen content fingerprints
    merged: list[Document] = []                        # Output list of unique documents

    for doc in bm25_docs + dense_docs:                 # Iterate over BM25 results first, then dense results
        fingerprint = doc.page_content.strip()[:200]   # Use the first 200 chars as a deduplication key (fast + effective)

        if fingerprint not in seen:                    # Only add documents that haven't seen yet
            seen.add(fingerprint)                      # Mark this fingerprint as seen
            merged.append(doc)                         # Add unique document to merged list

    return merged                                      # Return the deduplicated list


# ── Stage 3: Cross-encoder re-ranking ────────────────────────────────────────

def rerank_documents(
    query: str,
    docs: list[Document],
    top_n: int = TOP_N_RERANK
) -> list[tuple[Document, float]]:

    cross_encoder = _get_cross_encoder()               # Load (or retrieve cached) cross-encoder model

    # Build input pairs: each pair is (query, document_text)
    pairs = [                                          # List comprehension to create input pairs
        (query, doc.page_content)                      # Cross-encoder expects [query, passage] pairs
        for doc in docs                                # One pair per candidate document
    ]

    scores: list[float] = cross_encoder.predict(pairs).tolist()  # Score every (query, doc) pair in one batch call

    # Zip documents with their scores and sort best-first
    ranked = sorted(                                   # Sort the combined list ...
        zip(docs, scores),                             # ... of (Document, score) tuples ...
        key=lambda pair: pair[1],                      # ... by the score (second element of each tuple) ...
        reverse=True                                   # ... in descending order (highest score = most relevant)
    )

    # ── Print all scores for inspection ──────────────────────────────────────
    print(f"\n[Retrieval] ══════════ Reranking Scores ({len(ranked)} candidates) ══════════")
    for rank, (doc, score) in enumerate(ranked, start=1):              # Enumerate from 1 for human-friendly rank numbers
        source  = doc.metadata.get(                                    # Try to get a human-readable file name from metadata
            "file_name",                                               # LlamaIndex often stores this as 'file_name'
            doc.metadata.get("source", "unknown")                      # Fall back to 'source' or 'unknown'
        )
        snippet = doc.page_content[:90].replace("\n", " ")             # First 90 chars, newlines replaced with space
        marker  = " ◀ SELECTED" if rank <= top_n else ""              # Mark the chunks that will actually be used
        print(f"  Rank {rank:>2} | Score {score:+.4f} | {source}{marker}")  # Score (+ prefix shows sign clearly)
        print(f"         Snippet: {snippet}...")                       # Show a preview of the chunk content
    print(f"[Retrieval] ════════════════════════════════════════════════════════\n")

    top_docs = list(ranked)[:top_n]                    # Keep only the top-n documents after sorting

    print(f"[Retrieval] ✅ Returning top-{top_n} reranked document(s)")  # Log final selection

    return top_docs                                    # Return list of (Document, score) tuples


# ── Public pipeline entry point ───────────────────────────────────────────────

def retrieve(
    query: str,
    bm25_retriever: BM25Retriever,
    dense_retriever,
    top_n: int = TOP_N_RERANK
) -> list[tuple[Document, float]]:

    print(f"[Retrieval] Query: '{query}'")                              # Echo the query for debugging

    # ── Stage 1 & 2: Run both retrievers ─────────────────────────────────────
    bm25_docs  = bm25_retriever.invoke(query)                           # BM25 keyword search
    dense_docs = dense_retriever.invoke(query)                          # Dense vector search

    print(f"[Retrieval] BM25 returned  {len(bm25_docs)} doc(s)")        # Log BM25 hit count
    print(f"[Retrieval] Dense returned {len(dense_docs)} doc(s)")       # Log dense hit count

    # ── Stage 3: Merge and deduplicate ───────────────────────────────────────
    merged = _merge_results(bm25_docs, dense_docs)                      # Combine and deduplicate results
    print(f"[Retrieval] After merge: {len(merged)} unique doc(s)")      # Log merged count

    # ── Stage 4: Cross-encoder re-ranking ────────────────────────────────────
    ranked = rerank_documents(query, merged, top_n=top_n)               # Score and sort all candidates

    return ranked                                                        # Return final (Document, score) pairs
