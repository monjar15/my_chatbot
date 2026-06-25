from rag.embedding import load_chunks_cache           # Load cached chunks for BM25
from rag.embedding import load_vectorstore           # Load existing Qdrant vector store

from rag.retrieval import get_bm25_retriever         # BM25 retriever
from rag.retrieval import get_dense_retriever        # Dense retriever
from rag.retrieval import retrieve                   # Full retrieval pipeline


# =========================
# MAIN PROGRAM FUNCTION
# =========================
def main():

    print("\nLoading retrieval resources...")

    # =========================
    # LOAD EXISTING INDEXES
    # =========================
    chunks = load_chunks_cache()

    vectorstore = load_vectorstore()

    # =========================
    # BUILD RETRIEVERS
    # =========================
    bm25_retriever = get_bm25_retriever(
        chunks
    )

    dense_retriever = get_dense_retriever(
        vectorstore
    )

    print("Resources loaded successfully.")

    # =========================
    # CHAT LOOP
    # =========================
    while True:

        query = input(
            "\nEnter your search query (type 'exit' to stop): "
        ).strip()

        if query.lower() == "exit":
            print("Exiting program...")
            break

        if not query:
            print("Empty query. Try again.")
            continue

        try:

            print("\n" + "=" * 80)
            print(f"QUERY: {query}")
            print("=" * 80)

            # =========================
            # RETRIEVAL + RERANKING
            # =========================
            results = retrieve(
                query,
                bm25_retriever,
                dense_retriever
            )

            print(
                "\n===== FINAL RERANKED RESULTS =====\n"
            )

            for rank, (doc, score) in enumerate(
                results,
                start=1
            ):

                source = doc.metadata.get(
                    "file_name",
                    doc.metadata.get(
                        "source",
                        "unknown"
                    )
                )

                print(f"Rank: {rank}")

                print(f"Source: {source}")

                print(f"Rerank Score: {score:.4f}")

                print(
                    f"Text: {doc.page_content[:250]}"
                )

                print("-" * 60)

            print("\n" + "=" * 80)
            print("END OF RETRIEVAL TEST")
            print("=" * 80)

        except Exception as e:

            print(f"ERROR: {e}")

            print("\n" + "=" * 80)
            print("TEST FAILED")
            print("=" * 80)


# =========================
# PROGRAM ENTRY POINT
# =========================
if __name__ == "__main__":
    main()