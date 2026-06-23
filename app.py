# app.py

import os                                              
import streamlit as st                             

# ── Import our custom RAG modules ────────────────────────────────────────────
from rag.loader    import load_documents               # Step 1 – load raw files via LlamaIndex
from rag.embedding import (                            # Step 2 – split + embed + store
    split_documents,                                   # Split raw docs into chunks
    build_vectorstore,                                 # Embed chunks and store in Qdrant
    load_vectorstore,                                  # Load an existing Qdrant collection from disk
    load_chunks_cache                                  # Load saved text chunks for BM25
)
from rag.retrieval import (                            # Step 3 – retrieve relevant chunks
    get_bm25_retriever,                                # Build BM25 retriever from chunks
    get_dense_retriever,                               # Build dense vector retriever from Qdrant
    retrieve                                           # Run the full hybrid + rerank pipeline
)
from rag.chain import run_rag_chain_stream             # Step 4 – stream the LLM answer


# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_DOCS_FOLDER = "./documents"                    # Default folder users place their documents in


# ── Page configuration (must be the very first Streamlit call) ───────────────

st.set_page_config(                                    # Configure the Streamlit page metadata
    page_title="RAG Chatbot",                          # Browser tab title
    page_icon="🤖",                                    # Browser tab favicon emoji
    layout="wide"                                      # Use full browser width instead of narrow centre column
)


# ── Page header ──────────────────────────────────────────────────────────────

st.title("🤖 RAG Chatbot")                             # Large heading at the top of the page

st.divider()                                           # Horizontal line separating header from content


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR – Indexing controls and settings
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:                                       # Everything indented here appears in the left sidebar

    st.header("📁 Index Documents")                    # Sidebar section title

    # Text input for the folder path
    folder_path = st.text_input(                       # Single-line text box
        label="Documents folder",                      # Label shown above the text box
        value=DEFAULT_DOCS_FOLDER,                     # Pre-filled default value
        help="Path to a folder containing .txt, .pdf, .docx, .md files etc."  # Tooltip on hover
    )

    # ── Index button ──
    if st.button("🔄 Index Documents", use_container_width=True):   # Wide button; runs when clicked
        if not os.path.exists(folder_path):            # Validate that the folder exists before proceeding
            st.error(f"❌ Folder not found: `{folder_path}`")        # Show red error box
        else:
            # Loading step
            with st.spinner("📖 Loading documents..."):              # Spinner while loading
                docs = load_documents(folder_path)                   # Call loader.py to read all files
            st.info(f"📄 Loaded **{len(docs)}** document(s)")        # Show blue info box with doc count

            # Splitting & embedding step
            with st.spinner("✂️ Splitting and embedding chunks..."):  # Spinner while indexing
                chunks      = split_documents(docs)                  # Split raw docs into smaller chunks
                vectorstore = build_vectorstore(chunks)              # Embed and store chunks in Qdrant

            st.success(f"✅ Indexed **{len(chunks)}** chunks into Qdrant!")  # Green success message

            # Save everything to Streamlit session state so it persists across reruns
            st.session_state["indexed"]     = True         # Flag: documents have been indexed
            st.session_state["chunks"]      = chunks       # Cached chunks for BM25 retriever
            st.session_state["vectorstore"] = vectorstore  # Cached vectorstore for dense retriever

    st.divider()                                           # Visual separator in sidebar

    # ── Load existing index button ──
    st.caption("Already indexed? Load without re-processing:")         # Helper text above button

    if st.button("📂 Load Existing Index", use_container_width=True):  # Button to reload a previously built index
        with st.spinner("⏳ Loading existing index from disk..."):     # Spinner while loading
            try:
                vectorstore = load_vectorstore()                       # Load Qdrant collection from qdrant_storage/
                chunks      = load_chunks_cache()                      # Load BM25 chunk list from chunks_cache.pkl

                st.session_state["indexed"]     = True                 # Mark as indexed
                st.session_state["chunks"]      = chunks               # Store in session
                st.session_state["vectorstore"] = vectorstore          # Store in session

                st.success("✅ Existing index loaded successfully!")    # Confirm success

            except FileNotFoundError as e:                             # Handle missing files gracefully
                st.error(f"❌ {e}")                                    # Show specific error message

    st.divider()                                           # Visual separator

    # ── Retrieval settings ──
    st.header("⚙️ Settings")                               # Settings section header

    top_n = st.slider(                                     # Slider widget for choosing result count
        label="Top-N chunks after reranking",              # Slider label
        min_value=1,                                       # Minimum selectable value
        max_value=8,                                       # Maximum selectable value
        value=4,                                           # Default value
        help="How many reranked chunks are passed to the LLM as context."  # Tooltip
    )

    st.divider()                                           # Visual separator

    # ── Status indicator ──
    if st.session_state.get("indexed", False):             # Check if documents have been indexed
        chunk_count = len(st.session_state.get("chunks", []))  # Count cached chunks
        st.success(f"✅ Index ready ({chunk_count} chunks)")    # Green status pill
    else:
        st.warning("⚠️ No index loaded. Index documents first.")  # Yellow warning


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA – Chat interface
# ══════════════════════════════════════════════════════════════════════════════

# ── Initialise chat history in session state ──────────────────────────────────

if "messages" not in st.session_state:                     # Only initialise if the key doesn't exist yet
    st.session_state["messages"] = []                      # Start with an empty conversation history


# ── Render existing chat history ──────────────────────────────────────────────

for msg in st.session_state["messages"]:                   # Loop over every past message
    with st.chat_message(msg["role"]):                     # Render as 'user' or 'assistant' chat bubble
        st.markdown(msg["content"])                        # Render message text (supports markdown formatting)


# ── Chat input box ────────────────────────────────────────────────────────────

query = st.chat_input(                                     # Sticky input box pinned to the bottom of the page
    placeholder="Ask a question about your documents..."   # Grey placeholder text inside the input
)

# ── Handle new user query ─────────────────────────────────────────────────────

if query:                                                  # Only execute when the user actually submits a message

    # Guard: make sure documents are indexed before trying to answer
    if not st.session_state.get("indexed", False):         # Check session state flag
        st.warning("⚠️ Please index your documents first using the sidebar.")  # Remind the user
        st.stop()                                          # Stop further execution for this rerun

    # ── Display user message ──
    st.session_state["messages"].append(                   # Append user message to conversation history
        {"role": "user", "content": query}
    )
    with st.chat_message("user"):                          # Render user chat bubble
        st.markdown(query)                                 # Show the user's question

    # ── Retrieval phase ───────────────────────────────────────────────────────
    with st.chat_message("assistant"):                     # Open assistant chat bubble

        with st.spinner("🔍 Searching documents..."):      # Show spinner during retrieval (may take a few seconds)

            bm25_retriever  = get_bm25_retriever(          # Build BM25 retriever fresh each query (fast – in-memory)
                st.session_state["chunks"]                 # Pass the cached chunk list
            )
            dense_retriever = get_dense_retriever(         # Get dense retriever from the cached vectorstore
                st.session_state["vectorstore"]            # Pass the Qdrant-backed vectorstore
            )
            docs_with_scores = retrieve(                   # Run full pipeline: BM25 + dense + cross-encoder rerank
                query,                                     # User's question
                bm25_retriever,                            # BM25 keyword retriever
                dense_retriever,                           # Dense vector retriever
                top_n=top_n                                # From sidebar slider
            )

        # ── Show retrieved chunks in an expandable panel ──────────────────────

        with st.expander(                                  # Collapsible section (closed by default)
            f"📄 Retrieved Chunks & Scores  ·  top-{top_n} of {len(docs_with_scores)}",
            expanded=False                                 # Collapsed by default; user can click to expand
        ):
            for i, (doc, score) in enumerate(docs_with_scores, start=1):  # Iterate best-first
                source = doc.metadata.get(                 # Get source filename from metadata
                    "file_name",
                    doc.metadata.get("source", "unknown")
                )
                # Score badge: green if > 0, red if ≤ 0  (cross-encoder scores can be negative)
                score_color = "green" if score > 0 else "red"  # Choose badge colour based on score sign

                st.markdown(                               # Render chunk header with score badge
                    f"**Chunk {i}** &nbsp; "
                    f"🎯 Score: :{score_color}[`{score:+.4f}`] &nbsp; "
                    f"📁 `{source}`"
                )
                st.text(                                   # Show first 350 chars of chunk as plain text
                    doc.page_content[:350] + ("..." if len(doc.page_content) > 350 else "")
                )
                if i < len(docs_with_scores):              # Don't draw divider after the last chunk
                    st.divider()                           # Visual separator between chunks

        # ── Streaming LLM response ────────────────────────────────────────────

        response_placeholder = st.empty()                  # Create an empty container that we'll update in-place
        full_response        = ""                          # Accumulate all streamed tokens here

        for token in run_rag_chain_stream(query, docs_with_scores):  # Stream tokens from the LLM
            full_response += token                         # Append each new token to the full response string
            response_placeholder.markdown(                 # Update the displayed text on every token
                full_response + "▌"                        # Blinking cursor effect at the end
            )

        response_placeholder.markdown(full_response)       # Final render WITHOUT the cursor character

        # Append the completed assistant message to the conversation history
        st.session_state["messages"].append(               # Save assistant reply
            {"role": "assistant", "content": full_response}
        )
