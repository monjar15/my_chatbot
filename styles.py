# ── Sidebar styling (width, branding, buttons, search box, message actions) ──
SIDEBAR_CSS = """
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
"""

# ── App branding block shown at the top of the sidebar (static, no runtime values) ──
BRANDING_HTML = """\
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
"""

# ── Global app CSS: typeface, background, chat bubbles, chat input, sidebar lock ──
MAIN_CSS = """
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

/* Assistant Sources */
.askly-source-link,
.askly-source-link:link,
.askly-source-link:visited,
.askly-source-link:hover,
.askly-source-link:active {
    color: inherit !important;
    text-decoration: none !important;
    font-family: inherit !important;
    font-size: inherit !important;
    font-weight: inherit !important;
    cursor: pointer;
}

.askly-source-link:hover {
    opacity: 0.80;
}

.askly-source-link:focus,
.askly-source-link:focus-visible {
    outline: none !important;
    text-decoration: none !important;
}

.askly-source-text {
    color: inherit;
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
"""

# ── Stop button positioning CSS (hides the real button off-screen; the JS
#    proxy in app.py draws the visible ⏹ button in its place) ──
STOP_BUTTON_CSS = """
<style>
div[class*="st-key-askly_stop_btn_real"] {
    position: absolute !important;
    left: -9999px !important;
    top: -9999px !important;
}
</style>
"""

# ── Copy-to-clipboard iframe component template ──
# "__SAFE_JSON_TEXT__" is replaced in app.py with the JSON-escaped text to
# copy for a given message. A plain placeholder token is used (rather than
# str.format braces) so the CSS/JS curly braces below don't need doubling.
COPY_BUTTON_TEMPLATE = """
<html>
<head>
<style>
    html, body {
        margin: 0;
        padding: 0;
        background-color: transparent;
        display:flex;
        align-items:center;
        justify-content:center;
        height: 100%;
    }
    #copyBtn {
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
    }
    #copyBtn:hover {
        background: rgba(0,0,0,0.05);
        color: #2c3a3f;
        opacity: 1;
    }
</style>
</head>
<body>
    <button id="copyBtn" title="Copy">📋</button>
    <script>
        const btn = document.getElementById('copyBtn');
        const txt = __SAFE_JSON_TEXT__;

        btn.addEventListener('click', function() {
            function fallbackCopy(t) {
                const ta = document.createElement('textarea');
                ta.value = t;
                ta.style.position = 'fixed';
                ta.style.left = '-9999px';
                document.body.appendChild(ta);
                ta.focus();
                ta.select();
                try { document.execCommand('copy'); } catch (e) { console.error(e); }
                document.body.removeChild(ta);
            }

            if (navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(txt).catch(function() {
                    fallbackCopy(txt);
                });
            } else {
                fallbackCopy(txt);
            }

            const original = btn.innerText;
            btn.innerText = '✅';
            setTimeout(function() { btn.innerText = original; }, 1200);
        });
    </script>
</body>
</html>
"""