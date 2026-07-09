import os
import logging
from logging.handlers import RotatingFileHandler

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

LOG_DIR = "logs"                      # Directory where log files are stored
LOG_FILE = "chatbot.log"              # Log file name
LOG_LEVEL = logging.INFO              # Minimum level captured (INFO, WARNING, ERROR, CRITICAL)
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

MAX_LOG_SIZE_BYTES = 5 * 1024 * 1024  # Rotate once a log file hits 5 MB
BACKUP_COUNT = 3                      # Keep up to 3 old log files (chatbot.log.1, .2, .3)

# Third-party libraries whose verbose INFO logs we want suppressed
THIRD_PARTY_LOGGERS_TO_SILENCE = (
    "sentence_transformers",
    "huggingface_hub",
    "transformers",
    "httpx",
    "qdrant_client",
)

# ══════════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════════

os.makedirs(LOG_DIR, exist_ok=True)  # Ensure the logs directory exists before writing

logger = logging.getLogger("chatbot")
logger.setLevel(LOG_LEVEL)

# Guard against duplicate handlers if this module gets imported more than once
# (e.g. Streamlit reruns, or multiple modules importing `logger`)
if not logger.handlers:

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # ── Rotating file handler ────────────────────────────────────────────
    # Automatically rotates the log file once it exceeds MAX_LOG_SIZE_BYTES,
    # keeping BACKUP_COUNT old copies instead of growing one file forever.
    file_handler = RotatingFileHandler(
        filename=os.path.join(LOG_DIR, LOG_FILE),
        maxBytes=MAX_LOG_SIZE_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",   # Prevents UnicodeEncodeError from emoji/Unicode in log messages
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False  # Don't also send these logs up to the root logger

# ══════════════════════════════════════════════════════════════════════════
# THIRD-PARTY LOG NOISE REDUCTION
# ══════════════════════════════════════════════════════════════════════════

for lib_name in THIRD_PARTY_LOGGERS_TO_SILENCE:
    logging.getLogger(lib_name).setLevel(logging.WARNING)