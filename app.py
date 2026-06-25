import os
import streamlit as st
from datetime import datetime                          # for capturing timestamps

# ── Import our custom RAG modules ────────────────────────────────────────────
from rag.loader    import load_documents               # Step 1 – load raw files via LlamaIndex
from rag.cleaner   import clean_documents              # Step 2 - clean documents
from rag.embedding import (                            # Step 3 – split + embed + store
    split_documents,                                   # Split raw docs into chunks
    build_vectorstore,                                 # Embed chunks and store in Qdrant
    load_vectorstore,                                  # Load an existing Qdrant collection from disk
    load_chunks_cache                                  # Load saved text chunks for BM25
)
from rag.retrieval import (                            # Step 4 – retrieve relevant chunks
    get_bm25_retriever,                                # Build BM25 retriever from chunks
    get_dense_retriever,                               # Build dense vector retriever from Qdrant
    retrieve                                           # Run the full hybrid + rerank pipeline
)
from rag.chain import (
    run_rag_chain_stream                               # Step 5 – stream the LLM answer (normal RAG turn)
)
from rag.logger import logger                          # Shared logger instance


# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_DOCS_FOLDER = "./documents"                    # Default folder users place their documents in

# match the start of the assistant response to detect a "not found" turn.
NOT_FOUND_TRIGGER = "I wasn't able to find any results for"


# ── format a timestamp for display ────────────────────────────────────

def format_timestamp(dt: datetime) -> str:
    return dt.strftime("%I:%M %p · %b %d, %Y")           # e.g. 02:45 PM · Jun 25, 2026


# ── Page configuration (must be the very first Streamlit call) ───────────────

st.set_page_config(                                    # Configure the Streamlit page metadata
    page_title="RAG Chatbot",                          # Browser tab title
    page_icon="🤖",                                    # Browser tab favicon emoji
    layout="wide"                                      # Use full browser width instead of narrow centre column
)

if "app_started" not in st.session_state:              # Log startup only once per session

    logger.info(                                       # Record application startup
        "Chatbot application started"
    )

    st.session_state["app_started"] = True             # Prevent duplicate startup logs


# ── Page header ──────────────────────────────────────────────────────────────

st.title("🤖 Procedure Document Chatbot")             # Large heading at the top of the page

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

        logger.info(                                                 # Record indexing request
            f"Indexing started for folder: {folder_path}"
        )
                
        if not os.path.exists(folder_path):            # Validate that the folder exists before proceeding
            st.error(f"❌ Folder not found: `{folder_path}`")        # Show red error box
        else:
            # Loading step
            with st.spinner("📖 Loading documents..."):              # Spinner while loading
                docs = load_documents(folder_path)                   # Call loader.py to read all files

                logger.info(                                         # Record loaded document count
                    f"Loaded {len(docs)} document(s)"
                )

            st.info(f"📄 Loaded **{len(docs)}** document(s)")        # Show blue info box with doc count

            # Cleaning step
            with st.spinner("🧹 Cleaning documents..."):              # Spinner while cleaning
                docs = clean_documents(docs)                             # Clean extracted text

                logger.info(                                             # Record cleaning completion
                f"Cleaned {len(docs)} document(s)"
                )

            st.success(f"🧹 Cleaned **{len(docs)}** document(s)")    # Confirm text cleaning completed

            # Splitting & embedding step
            with st.spinner("✂️ Splitting and embedding chunks..."):  # Spinner while indexing
                chunks      = split_documents(docs)                  # Split raw docs into smaller chunks

                logger.info(                                         # Record chunk generation
                    f"Generated {len(chunks)} chunk(s)"
                )

                vectorstore = build_vectorstore(chunks)              # Embed and store chunks in Qdrant

                logger.info(                                         # Record vector storage
                    f"Stored {len(chunks)} vectors in Qdrant"
                )

            st.success(f"✅ Indexed **{len(chunks)}** chunks into Qdrant!")  # Green success message

            # Save everything to Streamlit session state so it persists across reruns
            st.session_state["indexed"]     = True         # Flag: documents have been indexed
            st.session_state["chunks"]      = chunks       # Cached chunks for BM25 retriever
            st.session_state["vectorstore"] = vectorstore  # Cached vectorstore for dense retriever

    st.divider()                                           # Visual separator in sidebar

    # ── Load existing index button ──
    st.caption("Already indexed? Load without re-processing:")         # Helper text above button

    if st.button("📂 Load Existing Index", use_container_width=True):  # Button to reload a previously built index

        logger.info(                                                    # Record index loading request
            "Loading existing index"
        )
                
        with st.spinner("⏳ Loading existing index from disk..."):     # Spinner while loading
            try:
                vectorstore = load_vectorstore()                       # Load Qdrant collection from qdrant_storage/
                chunks      = load_chunks_cache()                      # Load BM25 chunk list from chunks_cache.pkl

                st.session_state["indexed"]     = True                 # Mark as indexed
                st.session_state["chunks"]      = chunks               # Store in session
                st.session_state["vectorstore"] = vectorstore          # Store in session

                logger.info(                                           # Record successful index load
                    f"Loaded existing index ({len(chunks)} chunks)"
                )

                st.success("✅ Existing index loaded successfully!")    # Confirm success

            except FileNotFoundError as e:                             # Handle missing files gracefully
                st.error(f"❌ {e}")                                    # Show specific error message

    st.divider()                                           # Visual separator

    # ── Status indicator ──
    if st.session_state.get("indexed", False):             # Check if documents have been indexed
        chunk_count = len(st.session_state.get("chunks", []))  # Count cached chunks
        st.success(f"✅ Index ready ({chunk_count} chunks)")    # Green status pill
    else:
        st.warning("⚠️ No index loaded. Index documents first.")  # Yellow warning


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE – initialise all keys on first load
# ══════════════════════════════════════════════════════════════════════════════

if "messages" not in st.session_state:               # Only initialise if the key doesn't exist yet
    st.session_state["messages"] = []                # Full conversation history for display

if "awaiting_clarification" not in st.session_state:
    st.session_state["awaiting_clarification"] = False


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA – Chat interface
# ══════════════════════════════════════════════════════════════════════════════

for msg in st.session_state["messages"]:                    # Loop over every past message

    # Retrieve the saved timestamp string (fallback to empty string if somehow missing)
    ts = msg.get("timestamp", "")                           # pull saved timestamp

    if msg["role"] == "user":
        col1, col2 = st.columns([3, 1])                     # User occupies the LEFT 75% of the screen
        with col1:
            with st.chat_message("user"):                   # Avatar + bubble rendered in the left column
                st.markdown(msg["content"])
                if ts:                                      # show timestamp below message
                    st.caption(f"🕐 {ts}")

    else:
        col1, col2 = st.columns([1, 3])                     # Assistant occupies the RIGHT 75% of the screen
        with col2:
            with st.chat_message("assistant"):              # Avatar + bubble rendered in the right column
                st.markdown(msg["content"])
                if ts:                                      # show timestamp below message
                    st.caption(f"🕐 {ts}")


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

    # ── Capture the user's query timestamp the moment they submit ──────────────
    user_ts = format_timestamp(datetime.now())             # snapshot time of submission

    # ── Display user message (LEFT) ──
    st.session_state["messages"].append(                   # Append user message to conversation history
        {"role": "user", "content": query, "timestamp": user_ts}
    )

    logger.info(                                           # Record user query
        f"User query: {query}"
    )

    col1, col2 = st.columns([3, 1])                        # User bubble goes in the left column
    with col1:
        with st.chat_message("user"):                      # Render user chat bubble
            st.markdown(query)                             # Show the user's question
            st.caption(f"🕐 {user_ts}")                    # render user timestamp

    # ── Retrieval + assistant response (RIGHT) ─────────────────────────────────
    col1, col2 = st.columns([1, 3])                            # Assistant bubble goes in the right column
    with col2:
        with st.chat_message("assistant"):                     # Open assistant chat bubble

            response_placeholder = st.empty()                  # In-place container updated token-by-token
            full_response        = ""                          # Accumulator for the complete streamed reply

            # ══════════════════════════════════════════════════════════════════
            # BRANCH A — Clarification turn
            # The previous assistant message asked the user to reply Yes or No.
            # No retrieval is done — reply is determined by the user's typed answer.
            # ══════════════════════════════════════════════════════════════════

            if st.session_state["awaiting_clarification"]:

                logger.info(
                    f"Clarification reply received: {query}"
                )

                normalised = query.strip().lower()     # Normalise input for comparison

                if normalised == "yes":                # ── User confirmed they mistyped ──
                    full_response = (
                        "No problem! Please go ahead and type your corrected question, "
                        "and I'll search the documents again for you."
                    )
                    st.session_state["awaiting_clarification"] = False   # Reset flag; next turn is a fresh RAG query

                    logger.info(
                        "Clarification: user confirmed mistype → prompting re-query"
                    )

                elif normalised == "no":               # ── User confirmed the query was correct ──
                    full_response = (
                        "Thank you for confirming. Unfortunately, I was unable to find any "
                        "information on this topic in the company documents. "
                        "I recommend raising this concern with your direct supervisor or the "
                        "relevant department so they can assist you further."
                    )
                    st.session_state["awaiting_clarification"] = False   # Reset flag; conversation ends naturally

                    logger.info(
                        "Clarification: user confirmed query correct → escalation message sent"
                    )

                else:                                  # ── User typed something other than Yes / No ──
                    full_response = (
                        "I wasn't able to find any results for your query. "
                        "Please reply with **Yes** if you'd like to retype your question, "
                        "or **No** if the query was correct and you'd like to escalate it."
                    )
                    # Keep flag True so the next message is still routed here

                    logger.info(
                        "Clarification: unrecognised reply → re-prompting for Yes / No"
                    )

                response_placeholder.markdown(full_response)

                logger.info(
                    f"Clarification turn complete — awaiting_clarification={st.session_state['awaiting_clarification']}"
                )


            # ══════════════════════════════════════════════════════════════════
            # BRANCH B — Normal RAG turn
            # Run full retrieval + LLM pipeline, then check whether STEP 4
            # fired so we know whether to arm the clarification flag.
            # ══════════════════════════════════════════════════════════════════

            else:

                # Retrieve
                with st.spinner("🔍 Searching documents..."):      # Show spinner during retrieval (may take a few seconds)

                    bm25_retriever  = get_bm25_retriever(          # Build BM25 retriever fresh each query (fast – in-memory)
                    st.session_state["chunks"]                     # Pass the cached chunk list
                    )
                    dense_retriever = get_dense_retriever(         # Get dense retriever from the cached vectorstore
                    st.session_state["vectorstore"]                # Pass the Qdrant-backed vectorstore
                    )
                    docs_with_scores = retrieve(                   # Run full pipeline: BM25 + dense + cross-encoder rerank
                    query,                                     # User's question
                    bm25_retriever,                            # BM25 keyword retriever
                    dense_retriever                            # Dense vector retriever
                    )

                    logger.info(                                   # Record retrieval results
                        f"Retrieved {len(docs_with_scores)} reranked chunk(s)"
                    )

                # stream the LLM answer
                with st.spinner("💬 Generating response..."):
                    for token in run_rag_chain_stream(query, docs_with_scores):
                        full_response += token or ""
                        response_placeholder.markdown(full_response + "")

                response_placeholder.markdown(full_response)

                logger.info(
                    f"Generated response ({len(full_response)} chars)"
                )

                # ── display source filenames ──────────────────────────────
                # Extract unique source filenames from the retrieved chunks
                if not full_response.strip().startswith(NOT_FOUND_TRIGGER):     # Skip sources entirely if the LLM returned a "not found" reply
                    sources = sorted(set(                           # deduplicate and sort alphabetically
                        doc.metadata.get(                           # try LlamaIndex key first
                            "file_name",
                            doc.metadata.get("source", "unknown")  # fall back to 'source' or 'unknown'
                        )
                        for doc, _ in docs_with_scores             # iterate over all retrieved chunks
                    ))

                    if sources:                                    # only render if sources were found
                        source_list = "\n".join(f"- 📄 {s}" for s in sources)  # one bullet per file
                        st.caption(f"**Sources:**\n{source_list}")              # render below the answer

                # ── detect STEP 4 "not found" response ───────────────────
                # If the primary prompt could not find relevant context it will
                # open its reply with NOT_FOUND_TRIGGER.  We arm the flag so
                # the very next user message is routed to the clarification chain.
                if full_response.strip().startswith(NOT_FOUND_TRIGGER):
                    st.session_state["awaiting_clarification"] = True

                    logger.info(
                        "STEP 4 triggered — awaiting_clarification set to True"
                    )
                else:
                    # Normal answer found — ensure the flag stays False
                    st.session_state["awaiting_clarification"] = False

                # ── Shared: timestamp + append to history ─────────────────────────

            logger.info("=" * 80)

            assistant_ts = format_timestamp(datetime.now())
            st.caption(f"🕐 {assistant_ts}")

            st.session_state["messages"].append(
                {"role": "assistant", "content": full_response, "timestamp": assistant_ts}
            )
