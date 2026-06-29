import os
import streamlit as st
from datetime import datetime                          # for capturing timestamps
import uuid                                            # for generating unique chat IDs

# ── Import our custom RAG modules ────────────────────────────────────────────
from rag.loader    import load_documents               # Step 1 – load raw files
from rag.cleaner   import clean_documents              # Step 2 - clean documents
from rag.embedding import (                            # Step 3 – split + embed + store
    split_documents,                                   # Split raw docs into chunks
    build_vectorstore,                                 # Full index build (first run)
    add_documents_incremental,                         # Incremental index (new files only)
    load_vectorstore,                                  # Load an existing Qdrant collection from disk
    load_chunks_cache,                                 # Load saved text chunks for BM25
    load_recorder,                                     # Load file-path → mtime recorder from disk
    QDRANT_PATH,                                       # Path constant used to check if index exists
    CHUNKS_CACHE_PATH,                                 # Path constant used to check if cache exists
)
from rag.retrieval import (                            # Step 4 – retrieve relevant chunks
    get_bm25_retriever,                                # Build BM25 retriever from chunks
    get_dense_retriever,                               # Build dense vector retriever from Qdrant
    retrieve                                           # Run the full hybrid + rerank pipeline
)
from rag.chain import (
    run_rag_chain_stream                               # Step 5 – stream the LLM answer
)
from rag.logger import logger                          # Shared logger instance


# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_DOCS_FOLDER = "./documents"                    # Default folder users place their documents in

# match the start of the assistant response to detect a "not found" turn.
NOT_FOUND_TRIGGER = "I wasn't able to find any results for"
SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".ppt", ".pptm",      # Office docs
    ".md", ".txt",                                  # Plain text
    ".csv",                                         # Tabular
    ".epub", ".ipynb", ".hwp", ".mbox",             # Specialised
}

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── format a timestamp for display ────────────────────────────────────────────

def format_timestamp(dt: datetime) -> str:
    return dt.strftime("%I:%M %p · %b %d, %Y")           # e.g. 02:45 PM · Jun 25, 2026


# ── scan documents directory and record file metadata ─────────────────────────

def scan_documents_folder(folder: str) -> dict:

    file_mtimes = {}

    for root, _, files in os.walk(folder):             # Recurse through all subdirectories
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()   # Normalise extension to lowercase

            if ext in SUPPORTED_EXTENSIONS:            # Skip unsupported file types
                fpath = os.path.abspath(               # Always store absolute paths for reliable comparison
                    os.path.join(root, fname)
                )
                file_mtimes[fpath] = os.path.getmtime(fpath)   # Record last-modified timestamp

    return file_mtimes


# ── extract file path from document metadata ───────────────────────────────

def get_doc_filepath(doc) -> str:

    raw = doc.metadata.get(
        "file_path",
        doc.metadata.get(
            "source",
            doc.metadata.get("file_name", "")         # Last-resort fallback
        )
    )

    return os.path.abspath(raw) if raw else ""         # Normalise to absolute path


# ── Manage Vector Index Initialization and Incremental Updates ────────────

def run_startup_indexing(docs_folder: str, status_placeholder) -> tuple:

    # ── Validate documents folder ─────────────────────────────────────────
    if not os.path.exists(docs_folder):
        raise FileNotFoundError(
            f"Documents folder not found: '{docs_folder}'.\n\n"
            f"Please create the folder and add your documents "
            f"({', '.join(sorted(SUPPORTED_EXTENSIONS))}) before starting the app."
        )

    # ── Scan all supported files in the folder ─────────────────────────────
    current_files = scan_documents_folder(docs_folder)  # {abs_path: mtime}

    if not current_files:
        raise ValueError(
            f"No supported documents found in '{docs_folder}'.\n\n"
            f"Supported file types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    # ── Load existing recorder (empty dict on very first run) ──────────────
    recorder     = load_recorder()                      # {abs_path: mtime} of already-indexed files
    index_exists = (                                    # True only if BOTH artefacts are present on disk
        os.path.exists(QDRANT_PATH) and
        os.path.exists(CHUNKS_CACHE_PATH)
    )

    # ── Identify new or modified files ─────────────────────────────────────
    new_files = {
        fpath: mtime
        for fpath, mtime in current_files.items()
        if fpath not in recorder or recorder[fpath] != mtime
    }

    # ══════════════════════════════════════════════════════════════════════════
    # CASE A — First run: no index exists on disk → full build
    # ══════════════════════════════════════════════════════════════════════════

    if not index_exists:
        logger.info("First run detected — performing full index build")
        status_placeholder.info(
            f"📂 First run: building index from {len(current_files)} file(s)…"
        )

        # Load
        with st.spinner("📖 Loading documents..."):
            all_docs = load_documents(docs_folder)
            logger.info(f"Loaded {len(all_docs)} document(s)")

        # Clean
        with st.spinner("🧹 Cleaning documents..."):
            all_docs = clean_documents(all_docs)
            logger.info(f"Cleaned {len(all_docs)} document(s)")

        # Split
        with st.spinner("✂️ Splitting documents into chunks..."):
            chunks = split_documents(all_docs)
            logger.info(f"Generated {len(chunks)} chunk(s)")

        # Embed + store (passes current_files as the recorder to persist)
        with st.spinner("🧠 Embedding and storing vectors — this may take a few minutes..."):
            vectorstore = build_vectorstore(chunks, recorder=current_files)
            logger.info(f"Stored {len(chunks)} vectors in Qdrant")

        status_msg = (
            f"✅ Index built: {len(current_files)} file(s) · {len(chunks)} chunk(s)"
        )
        status_placeholder.success(status_msg)
        logger.info(status_msg)

        return vectorstore, chunks, status_msg

    # ══════════════════════════════════════════════════════════════════════════
    # CASE B — Subsequent run: new or modified files found → incremental update
    # ══════════════════════════════════════════════════════════════════════════

    elif new_files:
        logger.info(
            f"Incremental index: {len(new_files)} new/modified file(s) detected"
        )
        status_placeholder.info(
            f"🆕 {len(new_files)} new/modified file(s) detected. Updating index…"
        )

        # Load ALL docs from the folder (required by the loader's folder-based API),
        # then filter down to only the new/modified ones using metadata comparison.
        with st.spinner("📖 Loading documents..."):
            all_docs = load_documents(docs_folder)

        # Filter: keep only docs whose source file is in new_files
        new_docs = [
            doc for doc in all_docs
            if get_doc_filepath(doc) in new_files
        ]

        logger.info(
            f"Filtered to {len(new_docs)} new document object(s) "
            f"from {len(all_docs)} total loaded"
        )

        # Safety fallback: if metadata path matching returned zero results
        # (can happen if the loader stores relative paths or different keys),
        # log a warning and proceed with all loaded docs to avoid a silent
        # empty-index situation.  This is conservative — slightly over-indexes
        # rather than missing content.
        if not new_docs:
            logger.warning(
                "Metadata path filter matched 0 docs — "
                "falling back to full reload of all documents."
            )
            status_placeholder.warning(
                "⚠️ Could not isolate new files by metadata path. "
                "Re-indexing all documents as a safe fallback."
            )
            new_docs = all_docs

        # Clean
        with st.spinner("🧹 Cleaning new documents..."):
            new_docs = clean_documents(new_docs)

        # Split
        with st.spinner("✂️ Splitting new documents into chunks..."):
            new_chunks = split_documents(new_docs)
            logger.info(f"Generated {len(new_chunks)} new chunk(s)")

        # Append to existing Qdrant collection (does NOT recreate the collection)
        # Merge old recorder entries with the updated file mtimes.
        with st.spinner("🧠 Embedding and adding new vectors..."):
            updated_recorder = {**recorder, **new_files}   # Merge: old + new entries
            vectorstore = add_documents_incremental(new_chunks, updated_recorder)
            logger.info(f"Added {len(new_chunks)} new vectors to existing Qdrant collection")

        # Reload the full merged chunk cache that add_documents_incremental saved
        chunks = load_chunks_cache()

        status_msg = (
            f"✅ Index updated: +{len(new_files)} file(s) · "
            f"+{len(new_chunks)} new chunk(s) · "
            f"{len(chunks)} total chunk(s)"
        )
        status_placeholder.success(status_msg)
        logger.info(status_msg)

        return vectorstore, chunks, status_msg

    # ══════════════════════════════════════════════════════════════════════════
    # CASE C — Subsequent run, no changes → load existing index from disk
    # ══════════════════════════════════════════════════════════════════════════

    else:
        logger.info("No file changes detected — loading existing index from disk")
        status_placeholder.info("🔄 No new files detected. Loading existing index…")

        with st.spinner("🔄 Loading saved index..."):
            vectorstore = load_vectorstore()
            chunks      = load_chunks_cache()
            logger.info(f"Loaded existing index ({len(chunks)} chunk(s))")

        status_msg = (
            f"✅ Index loaded: {len(chunks)} chunk(s) · {len(recorder)} file(s)"
        )
        status_placeholder.success(status_msg)
        logger.info(status_msg)

        return vectorstore, chunks, status_msg


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
# SIDEBAR — Index status panel
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:

    # ── New Chat button ──
    if st.button("✏️ New Chat", use_container_width=True):  # Wide button; starts a fresh conversation
        new_id = str(uuid.uuid4())                          # Generate a unique ID for the new chat
        st.session_state["current_chat_id"] = new_id        # Set it as the active chat
        st.session_state["messages"] = []                   # Wipe displayed messages
        st.session_state["chat_history"] = []               # Wipe LLM memory
        st.session_state["awaiting_clarification"] = False  # Reset clarification flag
        st.session_state["chats"][new_id] = {               # Register the new chat in history
            "title": "New chat",                            # Placeholder title until first message
            "messages": [],                                 # Empty message list
            "timestamp": datetime.now()                     # Record creation time
        }
        logger.info(f"New chat started — id={new_id}")      # Log the new chat event
        st.rerun()                                          # Refresh UI to clear the chat area

    st.divider()  # Visual separator

    # ── Recent chats list ──
    st.caption("🕘 Recent chats")                           # Section label above the list
    if st.session_state.get("chats"):                       # Only render if there are past chats
        for chat_id, chat in sorted(                        # Iterate newest-first, skip empty chats
                [(k, v) for k, v in st.session_state["chats"].items() if v["messages"]],  # Skip chats with no messages
                key=lambda x: x[1]["timestamp"], reverse=True
        ):
            is_active = chat_id == st.session_state["current_chat_id"]  # Check if this is the active chat
            btn_label = f"▶ {chat['title']}" if is_active else chat["title"]  # Highlight active chat

            if st.button(btn_label, key=f"chat_{chat_id}", use_container_width=True):  # One button per chat
                st.session_state["current_chat_id"] = chat_id       # Switch to clicked chat
                st.session_state["messages"] = chat["messages"]     # Restore its messages
                st.session_state["chat_history"] = chat.get("chat_history", [])  # Restore LLM memory
                st.session_state["awaiting_clarification"] = False  # Reset clarification flag
                logger.info(f"Switched to chat id={chat_id}")       # Log chat switch
                st.rerun()                                          # Refresh UI with loaded chat

    st.divider()  # Visual separator

    st.header("📊 Index Status")
    # This placeholder is filled by run_startup_indexing() with progress messages,
    # and then overwritten with the final success/error status.
    index_status_placeholder = st.empty()

# ══════════════════════════════════════════════════════════════════════════════
# AUTO-INDEXING ON STARTUP
# Runs exactly once per session (guarded by the "indexed" session-state flag).
# Handles all three cases: first run, incremental update, and no-op load.
# ══════════════════════════════════════════════════════════════════════════════

if not st.session_state.get("indexed", False):

    try:
        vectorstore, chunks, status_msg = run_startup_indexing(
            docs_folder=DEFAULT_DOCS_FOLDER,
            status_placeholder=index_status_placeholder,
        )

        # Persist results in session state so subsequent reruns skip this block
        st.session_state["indexed"]      = True
        st.session_state["vectorstore"]  = vectorstore
        st.session_state["chunks"]       = chunks
        st.session_state["index_status"] = status_msg

    except (FileNotFoundError, ValueError) as e:
        # Expected configuration errors: folder missing, no supported files, etc.
        logger.error(f"Startup indexing failed: {e}")
        index_status_placeholder.error(f"❌ {e}")
        st.error(
            f"**Startup error:** {e}\n\n"
            "Please fix the issue and restart the app."
        )
        st.stop()                                      # Halt — no point showing the chat UI

    except Exception as e:
        # Unexpected errors (model download failure, Qdrant corruption, etc.)
        logger.error(f"Unexpected error during startup indexing: {e}", exc_info=True)
        index_status_placeholder.error(f"❌ Unexpected error: {e}")
        st.error(
            f"**Unexpected startup error:** {e}\n\n"
            "Check the terminal logs for details."
        )
        st.stop()

else:
    # Already indexed this session — just restore the status message in the sidebar
    index_status_placeholder.success(
        st.session_state.get("index_status", "✅ Index loaded")
    )


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE – Initialise all chat-related keys on first load
# ══════════════════════════════════════════════════════════════════════════════

if "messages" not in st.session_state:               # Only initialise if the key doesn't exist yet
    st.session_state["messages"] = []                # Full conversation history for display

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []            # plain memory list fed to the LLM prompt

if "awaiting_clarification" not in st.session_state:
    st.session_state["awaiting_clarification"] = False

if "chats" not in st.session_state:                  # Initialise chat history store on first load
    st.session_state["chats"] = {}                   # Dict of chat_id → {title, messages, timestamp}

if "current_chat_id" not in st.session_state:        # Track which chat is currently active
    st.session_state["current_chat_id"] = None       # None means no active chat yet

if not st.session_state["chats"] and st.session_state["current_chat_id"] is None:  # First ever load
    _init_id = str(uuid.uuid4())                             # Generate ID for the initial chat
    st.session_state["current_chat_id"] = _init_id          # Set as active
    st.session_state["chats"][_init_id] = {                 # Register in history
        "title":     "New chat",                            # Default title
        "messages":  [],                                    # Empty messages
        "timestamp": datetime.now()                         # Record creation time
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA – Chat interface
# This code is responsible for re-displaying all previous chat messages 
#    stored in Streamlit's session state whenever the app reruns.
# ══════════════════════════════════════════════════════════════════════════════

for msg in st.session_state["messages"]:                    # Loop over every past message

    # Retrieve the saved timestamp string (fallback to empty string if somehow missing)
    ts = msg.get("timestamp", "")                           # pull saved timestamp
    saved_sources = msg.get("sources", [])                  # retrieve stored list (empty = no sources)

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
                
                # Re-render source filenames saved at response time
                if saved_sources:                                # only render if sources exist
                    source_list = "\n".join(f"- 📄 {s}" for s in saved_sources)
                    st.caption(f"**Sources:**\n{source_list}")  # same format as the live render

                if ts:                                      # show timestamp below message
                    st.caption(f"🕐 {ts}")




# ── Chat input box ────────────────────────────────────────────────────────────

query = st.chat_input(                                     # Sticky input box pinned to the bottom of the page
    placeholder="Ask a question about your documents..."   # Grey placeholder text inside the input
)

# ── Handle new user query ─────────────────────────────────────────────────────

if query:                                                  # Only execute when the user actually submits a message

    # Guard — should never be False at this point (st.stop() above prevents it),
    # but kept as a defensive check.
    if not st.session_state.get("indexed", False):         # Check session state flag
        st.warning("⚠️ The index is not ready yet. Please wait for startup to complete.")  # Remind the user
        st.stop()                                          # Stop further execution for this rerun

    # ── Capture the user's query timestamp the moment they submit ──────────────
    user_ts = format_timestamp(datetime.now())             # snapshot time of submission

    # ── Append + display user message (LEFT) ──
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
            sources = []
            
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



            # ══════════════════════════════════════════════════════════════════
            # BRANCH B — Normal RAG turn
            # Run full retrieval + LLM pipeline, then check whether STEP 4
            # execute so we know whether to arm the clarification flag.
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
                    for token in run_rag_chain_stream(
                        query, 
                        docs_with_scores,
                        chat_history=st.session_state["chat_history"]   # pass memory
                    ):
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

                # ── Update LLM memory (only on clean RAG answers, not "not found") ────
                # We only append to memory when the LLM actually found something useful.
                # Clarification exchanges are intentionally excluded from memory to avoid
                # polluting history with "Yes / No" noise.
                if not full_response.strip().startswith(NOT_FOUND_TRIGGER):
                    st.session_state["chat_history"].append(             # Append user turn
                        {"role": "user", "content": query}
                    )
                    st.session_state["chat_history"].append(             # Append assistant turn
                        {"role": "assistant", "content": full_response}
                    )

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

            st.session_state["messages"].append({
                 "role": "assistant", 
                 "content": full_response, 
                 "timestamp": assistant_ts,
                 "sources": sources,
            })

            # ── Auto-title the chat using the first user message ──────────────────
            cid = st.session_state.get("current_chat_id")                                     # Get active chat ID
            if cid and st.session_state["chats"].get(cid, {}).get("title") == "New chat":     # Only rename on first message
                st.session_state["chats"][cid]["title"] = query[:40] + ("…" if len(query) > 40 else "")  # Use first query as title

            # ── Sync messages and memory back to chats history store ───────────────
            if cid := st.session_state.get("current_chat_id"):                                # Get active chat ID if set
                st.session_state["chats"][cid]["messages"]     = st.session_state["messages"]      # Persist messages
                st.session_state["chats"][cid]["chat_history"] = st.session_state["chat_history"]  # Persist LLM memory