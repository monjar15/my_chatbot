import os
import streamlit as st
from datetime import datetime                          # for capturing timestamps
import uuid                                            # for generating unique chat IDs
import html
import re
import textwrap
import json                                            # for safely escaping text into a JS string literal
import csv
from pathlib import Path

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

def get_doc_filepath(doc, docs_folder: str = DEFAULT_DOCS_FOLDER) -> str:

    raw = doc.metadata.get(
        "file_path",
        doc.metadata.get("source", "")
    )

    if raw:
        return os.path.abspath(raw)

    file_name = doc.metadata.get("file_name", "")
    return os.path.abspath(os.path.join(docs_folder, file_name)) if file_name else ""


# ── convert markdown bullet markers into a bullet glyph for display ────────

def format_bullets(text: str) -> str:
    """Replace markdown '- ' list markers with a bullet character for display."""
    return "\n".join(
        f"• {line.lstrip()[2:]}" if line.lstrip().startswith("- ") else line
        for line in text.split("\n")
    )

# ── wraps current text in the left-aligned bubble ────────

def render_bubble(placeholder, text="", show_timestamp=False, avatar_only=False,):

    if avatar_only:
        placeholder.markdown(
            """
            <div class="askly-row assistant">
                <div class="askly-avatar assistant">🤖</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    
    cleaned_text = text

    cleaned_text = re.sub(r"<[^>\n]*$", "", text)
    cleaned_text = re.sub(r"</?[^>]+>", "", cleaned_text)

    safe_text = html.escape(format_bullets(cleaned_text))
    safe_text = safe_text.replace("\n", "<br>")
    safe_text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe_text)  # match history redraw formatting

    meta_html = (
        f'<div class="askly-meta">🕐 {format_timestamp(datetime.now())}</div>'
        if show_timestamp else ""
    )

    bubble_html = (
        f'<div class="askly-row assistant">'
        f'<div class="askly-avatar assistant">🤖</div>'
        f'<div class="askly-content">'
        f'<div class="askly-bubble assistant">{safe_text}</div>'
        f'{meta_html}'
        f'</div>'
        f'</div>'
    )

    placeholder.markdown(bubble_html, unsafe_allow_html=True,)


# ── Feedback logging ──────────────────────────────────────────────────────
# Path to the CSV file that persists every feedback click across app restarts.
# session_state is memory-only and resets on refresh, so this file is the
# only durable record of user feedback.
FEEDBACK_LOG_PATH = "feedback_log.csv"


def _log_feedback_event(idx: int, msg: dict, feedback_value: str | None):
    """Append a single feedback event as a new row in the CSV log.

    Called every time a user clicks thumbs up/down (including un-clicking,
    which logs as 'cleared'). Creates the file with a header row on first use.
    """
    file_exists = Path(FEEDBACK_LOG_PATH).exists()

    with open(FEEDBACK_LOG_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Write header only once, when the file doesn't exist yet
        if not file_exists:
            writer.writerow([
                "timestamp",
                "chat_id",
                "message_index",
                "feedback",
                "assistant_response",
            ])

        writer.writerow([
            datetime.now().isoformat(),                          # when the click happened
            st.session_state.get("current_chat_id", ""),         # which chat this belongs to
            idx,                                                 # position of the message in the chat
            feedback_value if feedback_value else "cleared",     # "up" / "down" / "cleared"
            msg["content"][:200],                                # truncated preview of the assistant reply
        ])

    logger.info(f"Feedback logged: idx={idx}, value={feedback_value}")

def render_copy_button(idx: int, text: str):
    safe_json_text = json.dumps(text)

    copy_component_html = f"""
    <html>
    <head>
    <style>
        html, body {{
            margin: 0;
            padding: 0;
            background-color: transparent;
            display:flex;
            align-items:center;
            justify-content:center;
            height: 100%;
        }}
        #copyBtn {{
            height: 26px;
            width: 26px;
            padding: 0;
            border-radius: 6px;
            border: none;
            background: transparent;
            color: #8a939a;
            font-size: 13px;
            cursor: pointer;
            opacity: 0.85;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all .12s ease;
        }}
        #copyBtn:hover {{
            background: rgba(0,0,0,0.05);
            color: #2c3a3f;
            opacity: 1;
        }}
    </style>
    </head>
    <body>
        <button id="copyBtn" title="Copy">📋</button>
        <script>
            const btn = document.getElementById('copyBtn');
            const txt = {safe_json_text};

            btn.addEventListener('click', function() {{
                function fallbackCopy(t) {{
                    const ta = document.createElement('textarea');
                    ta.value = t;
                    ta.style.position = 'fixed';
                    ta.style.left = '-9999px';
                    document.body.appendChild(ta);
                    ta.focus();
                    ta.select();
                    try {{ document.execCommand('copy'); }} catch (e) {{ console.error(e); }}
                    document.body.removeChild(ta);
                }}

                if (navigator.clipboard && navigator.clipboard.writeText) {{
                    navigator.clipboard.writeText(txt).catch(function() {{
                        fallbackCopy(txt);
                    }});
                }} else {{
                    fallbackCopy(txt);
                }}

                const original = btn.innerText;
                btn.innerText = '✅';
                setTimeout(function() {{ btn.innerText = original; }}, 1200);
            }});
        </script>
    </body>
    </html>
    """

    st.iframe(copy_component_html, height=26, width=26)

# ── sync feedback edits back into the persistent chats dict ──────────────
def _sync_feedback_to_chat_store():
    cid = st.session_state.get("current_chat_id")
    if cid and cid in st.session_state["chats"]:
        st.session_state["chats"][cid]["messages"] = st.session_state["messages"]
    logger.info(f"Feedback synced to chat store — chat_id={cid}")


# ── remove an assistant reply and re-queue its query for regeneration ────
def _trigger_regeneration(idx: int):
    if idx == 0:
        return  # safety guard — an assistant msg should never be at index 0

    user_msg = st.session_state["messages"][idx - 1]

    # Drop the stale assistant turn from display history
    del st.session_state["messages"][idx]

    # Drop the matching pair from LLM memory, if present
    ch = st.session_state["chat_history"]
    if (
        len(ch) >= 2
        and ch[-1]["role"] == "assistant"
        and ch[-2]["role"] == "user"
        and ch[-2]["content"] == user_msg["content"]
    ):
        st.session_state["chat_history"] = ch[:-2]

    logger.info(f"Regenerate requested for query: {user_msg['content']}")

    st.session_state["regenerating"]  = True   # skip re-appending the user turn later
    st.session_state["pending_query"] = user_msg["content"]
    st.session_state["processing"]    = True
    st.rerun(scope="app")


# ── render the Copy / Feedback / Regenerate row under an assistant bubble ─
def render_message_actions(idx: int, msg: dict, is_last_assistant: bool):
    text = msg["content"]
    feedback = msg.get("feedback")

    st.markdown('<div class="askly-actions-anchor"></div>', unsafe_allow_html=True)

    # The key parameter renders as a class like "st-key-msg_actions_{idx}" on
    # this specific container. This lets CSS target ONLY this row instead of
    # every st.columns()/st.container(horizontal=True) in the whole app —
    # which was accidentally styling the sidebar's chat list buttons too.
    with st.container(horizontal=True, gap="small", key=f"msg_actions_{idx}"):
        render_copy_button(idx, text)

        if st.button("👍", key=f"fb_up_{idx}",
                      type="primary" if feedback == "up" else "secondary",
                      help="Give positive feedback"):
            msg["feedback"] = None if feedback == "up" else "up"
            _sync_feedback_to_chat_store()
            _log_feedback_event(idx, msg, msg["feedback"])
            st.rerun(scope="app")

        if st.button("👎", key=f"fb_down_{idx}",
                      type="primary" if feedback == "down" else "secondary",
                      help="Give negative feedback"):
            msg["feedback"] = None if feedback == "down" else "down"
            _sync_feedback_to_chat_store()
            _log_feedback_event(idx, msg, msg["feedback"])
            st.rerun(scope="app")

        if is_last_assistant:
            if st.button("🔄", key=f"regen_{idx}",
                          disabled=st.session_state["processing"],
                          help="Regenerate response"):
                _trigger_regeneration(idx)


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
                "Metadata path filter matched 0 docs for %d new/modified file(s) — "
                "falling back to a full rebuild instead of an incremental add "
                "to avoid duplicating existing vectors.",
                len(new_files),
            )
            status_placeholder.warning(
                "⚠️ Could not isolate new files by metadata path. "
                "Rebuilding the full index as a safe fallback."
            )
            all_docs = clean_documents(all_docs)
            chunks = split_documents(all_docs)
            vectorstore = build_vectorstore(chunks, recorder=current_files)  # full rebuild, not incremental append
            status_msg = f"✅ Index rebuilt (fallback): {len(current_files)} file(s) · {len(chunks)} chunk(s)"
            status_placeholder.success(status_msg)
            logger.info(status_msg)
            return vectorstore, chunks, status_msg

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
    page_title="Askly – Smart Answers, Anytime",       # Browser tab title
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
        if st.button("✏️ New Chat", key="new_chat_btn", use_container_width=True, disabled=busy,):  
            current_id = st.session_state.get("current_chat_id")
            current    = st.session_state["chats"].get(current_id)

            # If the active chat already has no messages, just reuse it instead
            # of registering another empty entry that will never be cleaned up.
            if current and not current["messages"]:
                st.session_state["messages"] = []
                st.session_state["chat_history"] = []
                st.session_state["awaiting_clarification"] = False
                st.rerun(scope="app")
            else:
                # Wide button; starts a fresh conversation
                new_id = str(uuid.uuid4())                          # Generate a unique ID for the  new chat
                st.session_state["current_chat_id"] = new_id        # Set it as the active chat
                st.session_state["messages"] = []                   # Wipe displayed messages
                st.session_state["chat_history"] = []               # Wipe LLM memory
                st.session_state["awaiting_clarification"] = False  # Reset clarification flag
                st.session_state["chats"][new_id] = {               # Register the new chat in history
                    "title": "New chat",                            # Placeholder title until first     message
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
                st.session_state["chats"][cid]["messages"] = []
                st.session_state["chats"][cid]["chat_history"] = []
                st.session_state["chats"][cid]["title"] = "New chat" # Reset title so next msg re-titles it
                st.session_state["chat_search_query"] = ""          # Reset search box so the cleared chat is visible in the list
            logger.info(f"Chat cleared — id={cid}")                 # Log the clear event
            st.rerun(scope="app")                                   # Refresh UI to clear the chat area

    st.divider()  # Visual separator

    # ══════════════════════════════════════════════════════════════════════════
    # ── Chat search box ──
    # Lives inside the fragment (not the outer app), so typing here reruns
    # only render_chat_sidebar() — NOT the main chat area — keeping it snappy.
    # Filters the "Recent chats" list below by matching against both the
    # chat title AND every message's content.
    # ══════════════════════════════════════════════════════════════════════════


    # ── Recent chats list ──
    st.markdown(
        "<span class='askly-recent-label'>🕘 Recent chats</span>",
        unsafe_allow_html=True,
    )

    search_query = st.text_input(  # Live-filter input box
        "Search chats",  # Accessibility label (hidden below)
        key="chat_search_query",  # Session-state key so we can reset it programmatically
        placeholder="🔍 Search chats...",  # Grey placeholder text shown when empty
        label_visibility="collapsed",  # Hide the label, keep only the placeholder
        disabled=busy,  # Lock the box while Askly is processing a query
    )

    # ── Recent chats section header (swaps label while actively searching) ──
    if search_query.strip():
        st.markdown(f"🔍 Results for “{search_query.strip()}”", unsafe_allow_html=True)  # Show what's being searched

    # ── Track which chat (if any) is currently being renamed ──
    if "renaming_chat_id" not in st.session_state:  # Only one chat can be in "rename mode" at a time
        st.session_state["renaming_chat_id"] = None

    recent_chats_container = st.container()
    
    with recent_chats_container:

        # ── Start with every non-empty chat (same base filter as before) ──
        all_chats = [
            (k, v) for k, v in st.session_state.get("chats", {}).items()
            if v["messages"]  # Skip chats with no messages
        ]

        # ── Apply the search filter, if the user has typed anything ──
        if search_query.strip():
            q = search_query.strip().lower()  # Normalise query for case-insensitive matching

            def _matches(chat: dict) -> bool:
                if q in chat["title"].lower():  # Match against the chat's title first (cheap check)
                    return True
                return any(  # Fall back to scanning every message's content
                    q in m.get("content", "").lower()
                    for m in chat["messages"]
                )

            all_chats = [(k, v) for k, v in all_chats if _matches(v)]  # Keep only chats that matched

        if all_chats:  # Only render if there's something to show
            for chat_id, chat in sorted(  # Iterate newest-first
                    all_chats,
                    key=lambda x: x[1]["timestamp"], reverse=True
            ):
                is_active = chat_id == st.session_state["current_chat_id"]  # Check if this is the active chat

                # ══════════════════════════════════════════════════════════
                # ── Rename mode for THIS chat: show text input + Save/Cancel ──
                # ══════════════════════════════════════════════════════════
                if st.session_state["renaming_chat_id"] == chat_id:

                    new_title = st.text_input(  # Editable title field, pre-filled with current title
                        "Rename chat",
                        value=chat["title"],
                        key=f"rename_input_{chat_id}",
                        label_visibility="collapsed",
                        max_chars=40,
                    )

                    save_col, cancel_col = st.columns([1, 1])  # Two equal-width buttons side by side

                    with save_col:
                        if st.button("💾 Save", key=f"save_rename_{chat_id}", use_container_width=True):
                            trimmed = (new_title or "").strip()  # Remove leading/trailing whitespace
                            if trimmed:  # Ignore empty titles — keep the old one
                                st.session_state["chats"][chat_id]["title"] = trimmed[:40] + (
                                    "…" if len(trimmed) > 40 else ""  # Same 40-char cap used for auto-titling
                                )
                                logger.info(f"Chat renamed — id={chat_id} → '{trimmed}'")  # Log the rename event
                            st.session_state["renaming_chat_id"] = None  # Exit rename mode
                            st.rerun(scope="app")  # Refresh so the new title shows everywhere

                    with cancel_col:
                        if st.button("✖ Cancel", key=f"cancel_rename_{chat_id}", use_container_width=True):
                            st.session_state["renaming_chat_id"] = None  # Exit rename mode, discard edits
                            st.rerun(scope="app")

                    # ── Enter = Save, Escape = Cancel ──
                    st.iframe(f"""
                    <script>
                    (function() {{
                        const doc = window.parent.document;

                        function attach() {{
                            const input     = doc.querySelector('.st-key-rename_input_{chat_id} input');
                            const saveBtn   = doc.querySelector('.st-key-save_rename_{chat_id} button');
                            const cancelBtn = doc.querySelector('.st-key-cancel_rename_{chat_id} button');

                            if (!input || !saveBtn || !cancelBtn) return false;
                            if (input.dataset.asklyBound) return true;   

                            input.dataset.asklyBound = "1";

                            input.addEventListener('keydown', function(e) {{
                                if (e.key === 'Enter') {{
                                    e.preventDefault();
                                    input.blur();                       
                                    setTimeout(function() {{            
                                        saveBtn.click();
                                    }}, 60);
                                }} else if (e.key === 'Escape') {{
                                        e.preventDefault();
                                        cancelBtn.click();
                                }}
                            }});

                            return true;
                        }}

                        if (!attach()) {{
                            const poller = setInterval(function() {{
                                if (attach()) clearInterval(poller);
                            }}, 150);
                            setTimeout(function() {{ clearInterval(poller); }}, 5000);
                        }}
                    }})();
                    </script>
                    """, height=1, width=1)

                # ══════════════════════════════════════════════════════════
                # ── Normal mode: chat button + small rename icon beside it ──
                # ══════════════════════════════════════════════════════════
                else:
                    chat_col, rename_col = st.columns([5, 1])  # Chat button takes most of the width; icon is narrow

                    with chat_col:
                        btn_label = f"▶ {chat['title']}" if is_active else chat["title"]  # Highlight active chat

                        if st.button(btn_label, key=f"chat_{chat_id}", use_container_width=True,
                                     disabled=busy, ):  # One    button per chat
                            st.session_state["current_chat_id"] = chat_id  # Switch to clicked chat
                            st.session_state["messages"] = chat["messages"]  # Restore its messages
                            st.session_state["chat_history"] = chat.get("chat_history", [])  # Restore LLM memory
                            st.session_state["awaiting_clarification"] = False  # Reset clarification flag
                            logger.info(f"Switched to chat id={chat_id}")  # Log chat switch
                            st.rerun(scope="app")  # Refresh whole app so main chat area updates too

                    with rename_col:
                        if st.button("⋮", key=f"rename_btn_{chat_id}", use_container_width=True,
                                     disabled=busy, ):  # Enter rename mode
                            st.session_state["renaming_chat_id"] = chat_id  # Mark this chat as being renamed
                            st.rerun(scope="app")  # Refresh to swap in the text input

        elif search_query.strip():  # Search typed but nothing matched
            st.caption("No matching chats found.")  # Empty-state message
        # else: no chats exist at all yet — render nothing, same as before

    st.write("")
    st.write("")

with st.sidebar:

    # ── Sidebar button styling ──────────────────────────────
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

    /* hover */
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
    
    /* ── Sidebar search box styling (matches dark sidebar theme) ── */
    section[data-testid="stSidebar"] div[data-testid="stTextInput"] input {
        background-color: #2a2a2a !important;   /* Dark input background to match sidebar */
        border: 1px solid #4a4a4a !important;   /* Slightly stronger border so the box itself reads clearly */
        border-radius: 8px !important;          /* Rounded corners, consistent with buttons */
        color: #f5f5f5 !important;              /* Bright text for whatever the user types */
        caret-color: #f5f5f5 !important;        /* Make sure the blinking cursor is visible too */
    }

    /* Placeholder text needs its own rule — browsers dim it heavily by default */
    section[data-testid="stSidebar"] div[data-testid="stTextInput"] input::placeholder {
        color: #b5b5b5 !important;              /* Light grey, but bright enough to read on #2a2a2a */
        opacity: 1 !important;                  /* Firefox dims placeholders further unless opacity is forced to 1 */
    }

    section[data-testid="stSidebar"] div[data-testid="stTextInput"] input:focus {
        border-color: #7a7a7a !important;       /* Lighter border on focus for feedback */
        box-shadow: none !important;            /* Remove Streamlit's default blue glow */
    }
    
    /* ── Message action row (Copy / Feedback / Regenerate) — ghost icon style ── */
    /* Scoped to containers created with key="msg_actions_{idx}" ONLY —
       using a bare stHorizontalBlock selector here was bleeding into the
       sidebar's chat-list st.columns(), squashing chat titles. */

    div[class*="st-key-msg_actions_"] button[data-testid="stBaseButton-secondary"],
    div[class*="st-key-msg_actions_"] button[data-testid="stBaseButton-primary"],
    div[class*="st-key-msg_actions_"] .stButton > button {
        height: 26px !important;
        width: 26px !important;
        min-height: 26px !important;
        min-width: 26px !important;
        padding: 0 !important;
        margin: 0 !important;
        border-radius: 6px !important;
        border: none !important;
        background: transparent !important;
        background-color: transparent !important;
        color: #8a939a !important;
        font-size: 13px !important;
        line-height: 1 !important;
        box-shadow: none !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        opacity: 0.85;
        transition: all .12s ease;
    }

    div[class*="st-key-msg_actions_"] button[data-testid="stBaseButton-secondary"]:hover,
    div[class*="st-key-msg_actions_"] button[data-testid="stBaseButton-primary"]:hover,
    div[class*="st-key-msg_actions_"] .stButton > button:hover {
        background: rgba(0,0,0,0.05) !important;
        background-color: rgba(0,0,0,0.05) !important;
        color: #2c3a3f !important;
        opacity: 1;
    }

    div[class*="st-key-msg_actions_"] button[data-testid="stBaseButton-secondary"]:active,
    div[class*="st-key-msg_actions_"] button[data-testid="stBaseButton-primary"]:active,
    div[class*="st-key-msg_actions_"] .stButton > button:active {
        background: rgba(0,0,0,0.09) !important;
        background-color: rgba(0,0,0,0.09) !important;
    }

    div[class*="st-key-msg_actions_"] button:disabled {
        opacity: .25 !important;
        background: transparent !important;
        background-color: transparent !important;
    }

    /* Selected feedback state — subtle darker tint, no filled pill */
    div[class*="st-key-msg_actions_"] button[data-testid="stBaseButton-primary"] {
        background: rgba(0,0,0,0.06) !important;
        background-color: rgba(0,0,0,0.06) !important;
        color: #2c3a3f !important;
        border: none !important;
        opacity: 1;
    }

    div[class*="st-key-msg_actions_"] button[data-testid^="stBaseButton"] p,
    div[class*="st-key-msg_actions_"] button[data-testid^="stBaseButton"] span {
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1 !important;
    }

    /* Copy button's iframe matches the same ghost footprint */
    div[class*="st-key-msg_actions_"] iframe {
        border: none !important;
        background: transparent !important;
        width: 26px !important;
        height: 26px !important;
        display: block !important;
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

if "stop_requested" not in st.session_state:         # True while the stop button has been clicked
    st.session_state["stop_requested"] = False       # Reset to False once handled

if "streaming_response" not in st.session_state:     # Durable copy of tokens streamed so far
    st.session_state["streaming_response"] = ""      # Survives the rerun triggered by clicking stop

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

assistant_indices = [i for i, m in enumerate(st.session_state["messages"]) if m["role"] == "assistant"]
last_assistant_idx = assistant_indices[-1] if assistant_indices else None

for idx, msg in enumerate(st.session_state["messages"]):
    # Retrieve the saved timestamp string (fallback to empty string if somehow missing)
    ts = msg.get("timestamp", "")                           # pull saved timestamp
    saved_sources = msg.get("sources", [])                  # retrieve stored list (empty =no  sources)
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
    if role == "assistant" and saved_sources:                # append sources inside theassistant  bubble block
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
    if role == "assistant" and not msg.get("stopped"):
        render_message_actions(idx, msg, is_last_assistant=(idx == last_assistant_idx))

# ── Stop button ──
if st.session_state["processing"]:
    with st.container(key="askly_stop_btn_real"):
        if st.button("Stop", key="askly_stop_btn"):
            st.session_state["stop_requested"] = True

    st.markdown("""
    <style>
    div[class*="st-key-askly_stop_btn_real"] {
        position: absolute !important;
        left: -9999px !important;
        top: -9999px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.iframe("""
        <script>
        (function() {
            function positionProxy() {
                const doc = window.parent.document;
                const realBtn  = doc.querySelector('div[class*="st-key-askly_stop_btn_real"] button');
                const arrowBtn = doc.querySelector('[data-testid="stChatInputSubmitButton"]');
                if (!realBtn || !arrowBtn) return;

                let proxy = doc.getElementById('askly-stop-proxy');
                if (!proxy) {
                    proxy = doc.createElement('button');
                    proxy.id = 'askly-stop-proxy';
                    proxy.innerText = '⏹';
                    proxy.style.position = 'fixed';
                    proxy.style.zIndex = '999999';
                    proxy.style.width = '34px';
                    proxy.style.height = '34px';
                    proxy.style.borderRadius = '8px';
                    proxy.style.border = 'none';
                    proxy.style.background = '#2c3a3f';
                    proxy.style.color = 'white';
                    proxy.style.fontSize = '14px';
                    proxy.style.cursor = 'pointer';
                    proxy.style.boxShadow = '0 1px 4px rgba(0,0,0,0.25)';
                    proxy.onmouseenter = () => proxy.style.background = '#445056';
                    proxy.onmouseleave = () => proxy.style.background = '#2c3a3f';
                    proxy.onclick = function() { realBtn.click(); };
                    doc.body.appendChild(proxy);
                }

                const rect = arrowBtn.getBoundingClientRect();
                proxy.style.top  = rect.top + 'px';
                proxy.style.left = (rect.left - 42) + 'px';
            }

            positionProxy();
            window.parent.addEventListener('resize', positionProxy);
            const poller = setInterval(positionProxy, 250);
            setTimeout(() => clearInterval(poller), 20000);
        })();
        </script>
        """, height=1, width=1)

else:
    st.iframe("""
    <script>
    (function() {
        const doc = window.parent.document;
        const proxy = doc.getElementById('askly-stop-proxy');
        if (proxy) proxy.remove();
    })();
    </script>
    """, height=1, width=1)


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

    # ── Handle stop-generation request ──────────────────────────────────
    if st.session_state.get("stop_requested"):                            # User clicked stop mid-generation
        partial = st.session_state.get("streaming_response", "")           # Recover whatever tokens made it through
        final_text = partial if partial.strip() else "Generation stopped."
        assistant_ts = format_timestamp(datetime.now())                    # Timestamp for the partial reply

        st.session_state["messages"].append({                              # Save the partial reply as a normal message
            "role": "assistant",
            "content": final_text,
            "timestamp": assistant_ts,
            "sources": [],
            "stopped": True,
        })

        logger.info("Generation stopped by user — partial response saved.")

        st.session_state["stop_requested"]     = False                     # Reset all flags back to idle state
        st.session_state["streaming_response"] = ""
        st.session_state["processing"]         = False
        st.session_state["pending_query"]      = None
        st.session_state["regenerating"]       = False

        st.rerun(scope="app")                                               # Refresh UI to unlock the input box
        st.stop()                                                           # Halt execution — skip the RAG logic below

    # Guard — should never be False at this point (st.stop() above prevents it),
    # but kept as a defensive check.
    if not st.session_state.get("indexed", False):         # Check session state flag
        st.warning("⚠️ The index is not ready yet. Please wait for startup to complete.")  # Remind the user
        st.stop()                                          # Stop further execution for this rerun

    # ── Capture the user's query timestamp the moment they submit ──────────────
    user_ts = format_timestamp(datetime.now())             # snapshot time of submission

    is_regenerating = st.session_state.get("regenerating", False)

    if not is_regenerating:
        # ── Append + display user message (LEFT) ──
        st.session_state["messages"].append(
            {"role": "user", "content": query, "timestamp": user_ts}
        )
        logger.info(f"User query: {query}")

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
    else:
        st.session_state["regenerating"] = False  # consume the flag
        logger.info(f"Regenerating answer for query: {query}")


    # ── Retrieval + assistant response (LEFT) ─────────────────────────────────

    response_placeholder = st.empty()                  # In-place container updated token-by-token
    full_response        = ""                          # Accumulator for the complete streamed reply
    sources = []
    
    render_bubble(
        response_placeholder,
        avatar_only=True
    )

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

        render_bubble(response_placeholder, full_response)


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

            # Build retrievers once per session and reuse — chunks/vectorstore
            # only change during startup indexing, not on every query.
            if "bm25_retriever" not in st.session_state:
                st.session_state["bm25_retriever"] = get_bm25_retriever(st.session_state["chunks"])
            if "dense_retriever" not in st.session_state:
                st.session_state["dense_retriever"] = get_dense_retriever(st.session_state["vectorstore"])

            bm25_retriever  = st.session_state["bm25_retriever"]
            dense_retriever = st.session_state["dense_retriever"]
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
                st.session_state["streaming_response"] = full_response  # persist tokens so a mid-stream Stop can recover them
                render_bubble(response_placeholder, full_response)
                if st.session_state.get("stop_requested"):               # bail out immediately instead of finishing the full stream
                    break

        render_bubble(response_placeholder, full_response, show_timestamp=True)

        logger.info(
            f"Generated response ({len(full_response)} chars)"
        )

        is_not_found = full_response.strip().startswith(NOT_FOUND_TRIGGER)

        # ── display source filenames ──────────────────────────────
        # Extract unique source filenames from the retrieved chunks
        if not is_not_found:                                # Skip sources entirely if the LLM returned a "not found" reply
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
        if not is_not_found:
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
        if is_not_found:
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
    if cid:                                                                                # Get ctive chat ID if set
        st.session_state["chats"][cid]["messages"]     = st.session_state["messages"]      # Persist messages
        st.session_state["chats"][cid]["chat_history"] = st.session_state["chat_history"]  # Persist LLM memory

    # ── Unlock the input box now that the assistant's reply is fully rendered ──
    st.session_state["processing"]    = False
    st.session_state["pending_query"] = None
    st.rerun(scope="app")