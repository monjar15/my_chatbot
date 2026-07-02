import os
import streamlit as st
from datetime import datetime                          # for capturing timestamps
import uuid                                            # for generating unique chat IDs
import html
import re
import textwrap

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
    run_rag_chain_stream,                              # Step 5 – stream the LLM answer
    rewrite_query_with_history                         # Resolve pronouns/follow-ups before retrieval
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


# ── convert markdown bullet markers into a bullet glyph for display ────────

def format_bullets(text: str) -> str:
    """Replace markdown '- ' list markers with a bullet character for display."""
    return "\n".join(
        f"• {line.lstrip()[2:]}" if line.lstrip().startswith("- ") else line
        for line in text.split("\n")
    )

# ── wraps current text in the left-aligned bubble ────────

def render_bubble(text, show_timestamp=False):

    clean_text = html.unescape(text)
    safe_text = html.escape(format_bullets(text)).replace("\n", "<br>")

    meta_html = (
        f'<div class="askly-meta">🕐 {format_timestamp(datetime.now())}</div>'
        if show_timestamp else ""
    )

    response_placeholder.markdown(
        f'<div class="askly-row assistant">'
        f'<div class="askly-avatar assistant">🤖</div>'
        f'<div class="askly-content">'
        f'<div class="askly-bubble assistant">{safe_text}</div>'
        f'{meta_html}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


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
    layout="wide",                                     # Use full browser width instead of narrow centre column
    initial_sidebar_state="expanded",
)

if "app_started" not in st.session_state:              # Log startup only once per session
    logger.info(                                       # Record application startup
        "Chatbot application started"
    )
    st.session_state["app_started"] = True             # Prevent duplicate startup logs


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Index status panel
# ══════════════════════════════════════════════════════════════════════════════
@st.fragment
def render_chat_sidebar():

    st.divider()  # Visual separator

    busy = st.session_state.get("processing", False)        # True while Askly is searching/generating

    button_container = st.container()

    with button_container:

    # ── New Chat button ──
        if st.button("✏️ New Chat", key="new_chat_btn", use_container_width=True, disabled=busy,):  # Wide button; starts a fresh conversation
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
            st.rerun(scope="app")                               # Refresh UI to clear the chat area

    # ── Clear Chat button ──
        if st.button("🧹 Clear Chat", key="clear_chat_btn", use_container_width=True ,disabled=busy,):  # Wide button; empties the    CURRENT chat (no new id)
            cid = st.session_state.get("current_chat_id")          # Get the currently active chat id
            st.session_state["messages"] = []                      # Wipe displayed messages
            st.session_state["chat_history"] = []                  # Wipe LLM memory
            st.session_state["awaiting_clarification"] = False     # Reset clarification flag
            if cid and cid in st.session_state["chats"]:            # Keep the chat entry, just empty it
                st.session_state["chats"][cid]["messages"]     = []
                st.session_state["chats"][cid]["chat_history"] = []
                st.session_state["chats"][cid]["title"]        = "New chat"  # Reset title so next msg re-titles it
            logger.info(f"Chat cleared — id={cid}")                 # Log the clear event
            st.rerun(scope="app")                                   # Refresh UI to clear the chat area

    st.divider()  # Visual separator

    # ── Recent chats list ──
    st.markdown(
        "<span class='askly-recent-label'>🕘 Recent chats</span>",
        unsafe_allow_html=True,
    )

    recent_chats_container = st.container()
    
    with recent_chats_container:
        if st.session_state.get("chats"):                       # Only render if there are past chats
            for chat_id, chat in sorted(                        # Iterate newest-first, skip empty chats
                    [(k, v) for k, v in st.session_state["chats"].items() if v["messages"]],  # Skip chats with     no messages
                    key=lambda x: x[1]["timestamp"], reverse=True
            ):
                is_active = chat_id == st.session_state["current_chat_id"]  # Check if this is the active chat
                btn_label = f"▶ {chat['title']}" if is_active else chat["title"]  # Highlight active chat

                if st.button(btn_label, key=f"chat_{chat_id}", use_container_width=True, disabled=busy,):  # One    button per chat
                    st.session_state["current_chat_id"] = chat_id       # Switch to clicked chat
                    st.session_state["messages"] = chat["messages"]     # Restore its messages
                    st.session_state["chat_history"] = chat.get("chat_history", [])  # Restore LLM memory
                    st.session_state["awaiting_clarification"] = False  # Reset clarification flag
                    logger.info(f"Switched to chat id={chat_id}")       # Log chat switch
                    st.rerun(scope="app")                               # Refresh whole app so main chat area updates too

    st.write("")
    st.write("")

with st.sidebar:

    # ── Sidebar button styling (ChatGPT-like) ──────────────────────────────
    st.markdown("""
    <style>
    
    /* Fixed sidebar width */
    [data-testid="stSidebar"]{
        width:320px !important;
        min-width:320px !important;
        max-width:320px !important;
    }

    [data-testid="stSidebar"][aria-expanded="true"]{
        width:320px !important;
        min-width:320px !important;
        max-width:320px !important;
    }

    [data-testid="stSidebar"] > div:first-child{
        width:320px !important;
        min-width:320px !important;
        max-width:320px !important;
    }
                
    /* Keep the sidebar content anchored to the top-left */
    [data-testid="stSidebarContent"]{
        display:flex;
        flex-direction:column;
        justify-content:flex-start;
        align-items:stretch;
        padding-top:0 !important;
    }

    /* Branding always starts from the upper-left corner */
    .askly-branding{
        display:flex;
        flex-direction:column;
        align-items:flex-start;
        justify-content:flex-start;
        gap:6px;
        width:100%;
        margin:0;
        padding:6px 4px 8px 4px;   /* smaller bottom spacing */
    }

    /* Divider between branding/tagline and New Chat button */
    [data-testid="stSidebar"] .askly-branding + div{
        margin-top:1em !important;
        padding-top:1em !important;
        border-top:1px solid #9ea7ad !important;
    }

    /* "Recent chats" label styling (replaces st.caption so we get a stable class hook) */
    .askly-recent-label{
        display:block;
        font-family:'Georgia','Times New Roman',serif !important;
        font-size:19.5px;          /* Increased from 13px by 50% */
        font-weight:600;           /* Slightly bolder for better emphasis */
        color:#4b5358;
        text-align:left;
        width:100%;
    }

    /* Invisible divider between "Recent chats" and the buttons */
    [data-testid="stSidebar"] .askly-recent-label + div{
        margin-top:8px !important;
        padding-top:8px !important;
        border-top:1px solid transparent !important;   /* invisible divider */
    }      

    /* New Chat / Clear Chat / Recent Chat buttons */
    /* Remove ALL spacing introduced by Streamlit wrappers */
    [data-testid="stSidebar"] .element-container,
    [data-testid="stSidebar"] div[data-testid="element-container"],
    [data-testid="stSidebar"] div[data-testid="stButton"],
    [data-testid="stSidebar"] .stButton{
        margin:0 !important;
        padding:0 !important;
    }

    /* Remove vertical gaps from every container */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"],
    [data-testid="stSidebar"] [data-testid="block-container"],
    [data-testid="stSidebar"] div[data-testid="stVerticalBlock"]{
        gap:0 !important;
        row-gap:0 !important;
    }

    /* Make consecutive widgets touch each other */
    /* Default spacing for consecutive buttons */
    [data-testid="stSidebar"] .element-container + .element-container,
    [data-testid="stSidebar"] div[data-testid="element-container"] + div[data-testid="element-container"]{
        margin-top:8px !important;      /* Increased from 6px → 8px (~33% increase, closest whole-pixel value) */
    }

    /* Keep New Chat and Clear Chat visually grouped with slightly larger separation */
    [data-testid="stSidebar"] button[key="new_chat_btn"],
    [data-testid="stSidebar"] button[key="clear_chat_btn"]{
        margin:0 !important;
    }

    /* Recent chat list gets a visible gap between buttons */
    [data-testid="stSidebar"] .askly-recent-label + div .element-container{
        margin-bottom:6px !important;
    }

    /* Button appearance */
    [data-testid="stSidebar"] .stButton > button{

        display:flex !important;
        align-items:center !important;
        justify-content:flex-start !important;

        width:100% !important;

        height:14px !important;
        min-height:14px !important;

        padding:0 6px !important;

        margin:0 !important;

        background:transparent !important;
        border:none !important;
        border-radius:6px !important;

        color:#2c3a3f !important;

        font-family:'Georgia','Times New Roman',serif !important;
        font-size:22.5px !important;      /* Changed from 15px */
        font-weight:600 !important;       /* Match New Chat/Clear Chat */
        line-height:1 !important;

        box-shadow:none !important;
        transition:background-color .12s ease;
    }

    /* First button = New Chat */
    /* Second button = Clear Chat */
    [data-testid="stSidebar"] .stButton:nth-of-type(1) > button,
    [data-testid="stSidebar"] .stButton:nth-of-type(2) > button{

        height:21px !important;
        min-height:21px !important;

        padding:0 8px !important;

        border-radius:8px !important;

        font-size:22.5px !important;
        font-weight:600 !important;
    }
                
    /* Remove all spacing around the caption */
    /* Prevent button captions from wrapping; truncate instead */
    [data-testid="stSidebar"] .stButton > button p,
    [data-testid="stSidebar"] .stButton > button span,
    [data-testid="stSidebar"] .stButton > button div{

        margin:0 !important;
        padding:0 !important;

        width:100% !important;

        line-height:1 !important;

        display:flex !important;
        align-items:center !important;
        justify-content:flex-start !important;

        text-align:left !important;

        white-space:nowrap !important;
        overflow:hidden !important;
        text-overflow:ellipsis !important;
    }

    /* ChatGPT-like hover */
    [data-testid="stSidebar"] .stButton > button:hover{
        background:rgba(255,255,255,.16) !important;
    }

    /* Active */
    [data-testid="stSidebar"] .stButton > button:active{
        background:rgba(255,255,255,.24) !important;
    }

    /* Focus */
    [data-testid="stSidebar"] .stButton > button:focus,
    [data-testid="stSidebar"] .stButton > button:focus-visible{
        outline:none !important;
        box-shadow:none !important;
    }

    /* Disabled */
    [data-testid="stSidebar"] .stButton > button:disabled{
        background:transparent !important;
        color:#7b848a !important;
        opacity:.6;
    }
                
    /* Remove internal spacing around the caption */
    /* Keep caption on a single line and truncate if necessary */
    [data-testid="stSidebar"] .stButton > button p,
    [data-testid="stSidebar"] .stButton > button span,
    [data-testid="stSidebar"] .stButton > button div {
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1 !important;
        width: 100% !important;
        text-align: left !important;
        justify-content: flex-start !important;

        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    
    /* Force the caption itself to stay left-aligned */
    [data-testid="stSidebar"] .stButton > button p,
    [data-testid="stSidebar"] .stButton > button span,
    [data-testid="stSidebar"] .stButton > button div {
        width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
        text-align: left !important;
        justify-content: flex-start !important;
    }
    
    /* Hover */
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255,255,255,.18) !important;
    }
    
    /* Click */
    [data-testid="stSidebar"] .stButton > button:active {
        background: rgba(255,255,255,.28) !important;
    }
    
    /* Focus */
    [data-testid="stSidebar"] .stButton > button:focus,
    [data-testid="stSidebar"] .stButton > button:focus-visible {
        outline: none !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Disabled */
    [data-testid="stSidebar"] .stButton > button:disabled {
        background: transparent !important;
        color: #7b848a !important;
        opacity: .6;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── App branding ──
    branding_html = textwrap.dedent("""\
    <div class="askly-branding">
        <div style="display:flex; align-items:center; gap:12px;">
            <svg width="48" height="48" viewBox="0 0 200 200">
                <defs>
                    <linearGradient id="asklyGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stop-color="#37474f"></stop>
                        <stop offset="100%" stop-color="#607d8b"></stop>
                    </linearGradient>
                </defs>
                <path d="M100 20C58.6 20 25 50 25 87.5c0 23.3 11.3 44.2 29.2 57.9V178l33.3-17.9c4.5 0.6 9 0.9 12.5 0.9 41.4 0 75-30 75-67.5S141.4 20 100 20z"
                      fill="url(#asklyGrad)"></path>
                <circle cx="71" cy="87" r="10.5" fill="white"></circle>
                <circle cx="100" cy="87" r="10.5" fill="white"></circle>
                <circle cx="129" cy="87" r="10.5" fill="white"></circle>
            </svg>
            <span style="font-family:'Georgia','Times New Roman',serif; font-size:42px; font-weight:700; letter-spacing:-0.5px; color:#2c3a3f;">
                Askly
            </span>
        </div>
        <span style="font-family:'Georgia','Times New Roman',serif; font-size:13px; color:#4b5358; text-align:left; width:100%;">
            Ask confidently. Find accurately. Askly.
        </span>
    </div>
    """)

    st.markdown(branding_html, unsafe_allow_html=True)

    render_chat_sidebar()

# ── Silent status sink ────────────────────────────────────────────────────
# No longer displayed in the sidebar. run_startup_indexing() still calls
# .info()/.success()/.error()/.warning() on this object internally, so we
# give it no-op methods instead of removing the variable entirely.
class _SilentStatus:
    def info(self, *args, **kwargs):    pass
    def success(self, *args, **kwargs): pass
    def error(self, *args, **kwargs):   pass
    def warning(self, *args, **kwargs): pass

index_status_placeholder = _SilentStatus()

# ── Display the welcome title immediately while startup indexing runs ──
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if (
    len(st.session_state["messages"]) == 0
    and st.session_state.get("pending_query") is None
):
    st.title("What can Askly help you today?")
    st.divider()


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
    pass


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE – Initialise all chat-related keys on first load
# ══════════════════════════════════════════════════════════════════════════════

if "messages" not in st.session_state:               # Only initialise if the key doesn't exist yet
    st.session_state["messages"] = []                # Full conversation history for display

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []            # plain memory list fed to the LLM prompt

if "awaiting_clarification" not in st.session_state:
    st.session_state["awaiting_clarification"] = False

if "processing" not in st.session_state:            # True while a query is being answered
    st.session_state["processing"] = False

if "pending_query" not in st.session_state:          # Holds the query between the submit-rerun and the answer-rerun
    st.session_state["pending_query"] = None

if "chats" not in st.session_state:                  # Initialise chat history store on first load
    st.session_state["chats"] = {}                   # Dict of chat_id → {title, messages, timestamp}

if "current_chat_id" not in st.session_state:        # Track which chat is currently active
    st.session_state["current_chat_id"] = None       # None means no active chat yet

if not st.session_state["chats"] and st.session_state["current_chat_id"] is None:  # First ever load
    _init_id = str(uuid.uuid4())                            # Generate ID for the initial chat
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

# ── Chat bubble CSS (true left/right alignment via flexbox, not columns) ──
st.markdown(
    """
    <style>
    /* Apply a consistent, formal serif typeface across the ENTIRE app,
       including elements Streamlit renders via Emotion CSS-in-JS
       (buttons, captions, widget labels) that the old selector missed. */
    html, body, [class*="css"], [data-testid="stAppViewContainer"],
    [data-testid="stSidebar"], [data-testid="stSidebarContent"],
    [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"],
    [data-testid="stChatInput"], .stMarkdown, .stChatInput, .stButton,
    .stCaption, button, button p, button span, button div,
    [data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-secondary"] p,
    [data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primary"] p,
    input, textarea, label, p, span, div {
        font-family: 'Georgia', 'Times New Roman', serif !important;
    }

    /* Main display area — darker formal slate-gray tone */
    [data-testid="stAppViewContainer"] > .main {
        background-color: #c4c9ce;
    }
    /* Sidebar — complementary, slightly lighter/cooler slate tone */
    [data-testid="stSidebar"] {
        background-color: #b6bdc4;
    }
    .askly-row { display: flex; margin: 6px 0; align-items: flex-end; gap: 8px; }
    .askly-row > div { 
        display: flex; 
        flex-direction: column; 
        flex: 0 1 auto; 
        min-width: 0;
        max-width: 75%; 
    }
    .askly-row.user > div { align-items: flex-end; }
    .askly-row.assistant > div { align-items: flex-start; }
    .askly-row.user { justify-content: flex-end; }       /* user → right */
    .askly-row.assistant { justify-content: flex-start; } /* assistant → left */

    .askly-content {
        display: inline-flex;
        flex-direction: column;
        width: auto;
        max-width: 75%;
        flex: 0 1 75%;
    }

    .askly-avatar {
        width: 28px;
        height: 28px;
        min-width: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 15px;
        flex-shrink: 0;
    }
    .askly-avatar.user {
    background: linear-gradient(90deg,#37474f,#607d8b);
    }
    .askly-avatar.assistant {
    background: linear-gradient(90deg,#dbe1e4,#c2c9ce);
    }

    .askly-bubble {
        display: inline-block;          /* Size naturally to the text */
        width: auto;             
        max-width: 100%;                 /* Wrap only after reaching 75% of the chat width */

        box-sizing: border-box;
        padding: 10px 16px;
        border-radius: 16px;
        font-size: 15px;
        line-height: 1.5;

        white-space: pre-wrap;          /* Preserve user line breaks */
        overflow-wrap: anywhere;      /* Wrap only when necessary */
        word-break: break-word;
    }

    .askly-bubble.user {
        background: linear-gradient(90deg,#37474f,#607d8b);  /* matches Askly logo gradient */
        color: #f5f7f8;
        border-bottom-right-radius: 4px;
    }
    .askly-bubble.assistant {
    /* lighter tint of the same slate hue, instead of indigo/cyan */
        background: linear-gradient(90deg,#e7eaec,#d4dade);
        color: #2c3a3f;
        border: 1px solid #c2c9ce;
        border-bottom-left-radius: 4px;
    }

    .askly-meta { 
        font-size: 11px; 
        opacity: 0.65; 
        margin-top: 4px; 
        width: 100%; 
        align-self: stretch; 
    }

    .askly-row.user .askly-meta { text-align: right; }
    .askly-row.assistant .askly-meta { text-align: right; }

    .askly-sources {
        font-size: 11px;
        opacity: 0.65;
        margin-top: 4px;
        width: 100%;
        align-self: stretch;
        text-align: left;
    }

    /* Force the sidebar to remain expanded */
    [data-testid="stSidebar"]{
        transform: none !important;
        visibility: visible !important;
    }

    /* Hide ALL collapse / expand controls */
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarCollapseButton"]{
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
        overflow: hidden !important;
    }

    /* Prevent Streamlit from reserving space for the hidden button */
    [data-testid="stSidebarNav"]{
        padding-top: 0 !important;
    }

    /* Chat input styling (matches Askly gray/slate theme)          */
    /* Remove the OUTER Streamlit box */
    [data-testid="stChatInput"] {
        position: relative !important;
    }

    [data-testid="stChatInput"] > div {
        background: transparent !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }

    /* Remove focus ring from outer container */
    [data-testid="stChatInput"] > div:focus-within {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }

    /* Style only the actual input */
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] input {
        background: #eef1f3 !important;
        color: #2c3a3f !important;
        border: 1px solid #9ea7ad !important;
        border-radius: 12px !important;
        box-shadow: none !important;
        min-height: 1.5em !important;   /* 1.5x text size for a sleeker, classier box */
        line-height: 1.5 !important;
        font-size: 16px !important;
        padding: 14px 52px 14px 14px !important;   /* room on the right for the button */
        width: 100% !important;
        display: block !important;
    }

    /* Input focus */
    [data-testid="stChatInput"] textarea:focus,
    [data-testid="stChatInput"] textarea:focus-visible,
    [data-testid="stChatInput"] input:focus,
    [data-testid="stChatInput"] input:focus-visible {
        border: 1px solid #607d8b !important;
        outline: none !important;
        box-shadow: none !important;
    }

    /* Placeholder */
    [data-testid="stChatInput"] textarea::placeholder,
    [data-testid="stChatInput"] input::placeholder {
        color: #6d777d !important;
        opacity: 1;
    }

    /* Send button */
    [data-testid="stChatInputSubmitButton"] {
        background: linear-gradient(90deg,#37474f,#607d8b) !important;
        border: none !important;
        border-radius: 8px !important;
        color: white !important;
        box-shadow: none !important;
        position: absolute !important;
        right: 8px !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        width: 34px !important;
        height: 34px !important;
        min-width: 34px !important;
        z-index: 2 !important;
    }

    /* Hover */
    [data-testid="stChatInput"] button:hover {
        background: linear-gradient(90deg,#455a64,#78909c) !important;
    }

    /* Button focus */
    [data-testid="stChatInput"] button:focus,
    [data-testid="stChatInput"] button:focus-visible {
        outline: none !important;
        box-shadow: none !important;
    }

    /* Arrow icon */
    [data-testid="stChatInput"] button svg {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

for msg in st.session_state["messages"]:                    # Loop over every past message

    # Retrieve the saved timestamp string (fallback to empty string if somehow missing)
    ts = msg.get("timestamp", "")                           # pull saved timestamp
    saved_sources = msg.get("sources", [])                  # retrieve stored list (empty = no sources)
    role = msg["role"]                                      # "user" or "assistant"
    avatar_icon = "🧑" if role == "user" else "🤖"           # icon shown per role

    if role == "assistant":
        content_html = html.escape(format_bullets(html.unescape(msg["content"])))
        content_html = content_html.replace("\n", "<br>")
        # Convert markdown **bold** to real <b> tags now that text is escaped
        content_html = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", content_html)
    else:
        content_html = html.escape(msg["content"])    
    
    bubble_html = f'<div class="askly-bubble {role}">{content_html}</div>'

    if ts:
        bubble_html += f'<div class="askly-meta">🕐 {ts}</div>'

    if role == "assistant" and saved_sources:                # append sources inside the assistant bubble block
        source_list = "<br>".join(f"📄 {s}" for s in saved_sources)
        bubble_html += f'<div class="askly-sources"><b>Sources:</b><br>{source_list}</div>'

    avatar_html = f'<div class="askly-avatar {role}">{avatar_icon}</div>'

    # user → bubble first, avatar after (avatar sits on the right);
    # assistant → avatar first, bubble after (avatar sits on the left)
    if role == "user": 
        row_inner = f'<div class="askly-content">{bubble_html}</div>{avatar_html}'
    else:
        row_inner = f'{avatar_html}<div class="askly-content">{bubble_html}</div>'

    st.markdown(
    f'<div class="askly-row {role}">{row_inner}</div>',
    unsafe_allow_html=True,
    )


# ── Chat input box ────────────────────────────────────────────────────────────

query = st.chat_input(                                      # Sticky input box pinned to the bottom of the page
    placeholder="Your answer starts with Askly...",         # Grey placeholder text inside the input
    disabled=st.session_state["processing"],                # Lock the box while a response is being generated
) 

if query and not st.session_state["processing"]:            # Fresh submission → arm processing flag and rerun
    st.session_state["processing"] = True                   # This makes the disabled box render BEFORE the RAG work starts
    st.session_state["pending_query"] = query
    st.rerun(scope="app")

# ── Handle new user query ─────────────────────────────────────────────────────

if st.session_state["processing"] and st.session_state["pending_query"]:  # Only execute on the follow-up rerun
    query = st.session_state["pending_query"]                             # Recover the query saved before the rerun

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

    st.markdown(
        f'<div class="askly-row user">'
        f'<div class="askly-content">'
        f'<div class="askly-bubble user">{query}</div>'
        f'<div class="askly-meta">🕐 {user_ts}</div>'
        f'</div>'
        f'<div class="askly-avatar user">🧑</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Retrieval + assistant response (LEFT) ─────────────────────────────────

    response_placeholder = st.empty()                  # In-place container updated token-by-token
    full_response        = ""                          # Accumulator for the complete streamed reply
    sources = []
            
    render_bubble(full_response)                        # Show avatar + empty bubble immediately (covers spinner phases)

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

        render_bubble(full_response)


    # ══════════════════════════════════════════════════════════════════
    # BRANCH B — Normal RAG turn
    # Run full retrieval + LLM pipeline, then check whether STEP 4
    # execute so we know whether to arm the clarification flag.
    # ══════════════════════════════════════════════════════════════════

    else:

        # Retrieve
        with st.spinner("🔍 Searching documents..."):      # Show spinner during retrieval (may take a few seconds)

            search_query = rewrite_query_with_history(     # Resolve pronouns/follow-ups BEFORE retrieval runs
            query,                                         # Original user question (raw)
            st.session_state["chat_history"]               # Prior turns used to resolve references
            )
            bm25_retriever  = get_bm25_retriever(          # Build BM25 retriever fresh each query (fast – in-memory)
            st.session_state["chunks"]                     # Pass the cached chunk list
            )
            dense_retriever = get_dense_retriever(         # Get dense retriever from the cached vectorstore
            st.session_state["vectorstore"]                # Pass the Qdrant-backed vectorstore
            )
            docs_with_scores = retrieve(                   # Run full pipeline: BM25 + dense + cross-encoder rerank
            search_query,                                  # User's question
            bm25_retriever,                                # BM25 keyword retriever
            dense_retriever                                # Dense vector retriever
            )

            logger.info(                                   # Record retrieval results
                f"Retrieved {len(docs_with_scores)} reranked chunk(s)"
            )

        # stream the LLM answer
        with st.spinner("💬 Generating response..."):
            for token in run_rag_chain_stream(
                search_query, 
                docs_with_scores,
                chat_history=st.session_state["chat_history"]   # pass memory
            ):
                full_response += token or ""
                render_bubble(full_response + "")

        render_bubble(full_response, show_timestamp=True)

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
                source_list = "<br>".join(f"📄 {s}" for s in sources)  # one bullet per file
                st.markdown(f'<div class="askly-meta"><b>Sources:</b><br>{source_list}</div>',
                        unsafe_allow_html=True,
                )                                                       # render below the answer

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

    st.session_state["messages"].append({
         "role": "assistant", 
         "content": full_response, 
         "timestamp": assistant_ts,
         "sources": sources,
    })

    # ── Auto-title the chat using the first user message ──────────────────
    cid = st.session_state.get("current_chat_id")                                     # Get ctive chat ID
    if cid and st.session_state["chats"].get(cid, {}).get("title") == "New chat":     # nly rename on first message
        st.session_state["chats"][cid]["title"] = query[:40] + ("…" if len(query) > 40 else "")  # Use first query as title

    # ── Sync messages and memory back to chats history store ───────────────
    if cid := st.session_state.get("current_chat_id"):                                # Get ctive chat ID if set
        st.session_state["chats"][cid]["messages"]     = st.session_state["messages"]      # Persist messages
        st.session_state["chats"][cid]["chat_history"] = st.session_state["chat_history"]  # Persist LLM memory

    # ── Unlock the input box now that the assistant's reply is fully rendered ──
    st.session_state["processing"]    = False
    st.session_state["pending_query"] = None
    st.rerun(scope="app")