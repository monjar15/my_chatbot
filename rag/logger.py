import os                                           # Standard library for file and directory operations

import logging                                      # Python logging framework


# ── Logging Configuration ────────────────────────────────────────────────────

LOG_DIR = "logs"                                    # Directory where log files will be stored

LOG_FILE = "chatbot.log"                            # Name of the log file

LOG_LEVEL = logging.INFO                            # Record INFO, WARNING, ERROR and CRITICAL logs


# ── Create Log Directory ─────────────────────────────────────────────────────

os.makedirs(                                        # Create logs directory if it does not exist
    LOG_DIR,                                        # Target directory path
    exist_ok=True                                   # Prevent error if directory already exists
)


# ── Create Shared Logger ─────────────────────────────────────────────────────

logger = logging.getLogger(                         # Create named logger instance
    "chatbot"                                       # Logger name
)

logger.setLevel(                                    # Set minimum logging level
    LOG_LEVEL                                       # INFO and above
)


# ── Prevent Duplicate Handlers ───────────────────────────────────────────────

if not logger.handlers:                             # Configure logger only once

    file_handler = logging.FileHandler(             # Create file output handler
        os.path.join(                               # Build full log file path
            LOG_DIR,                                # Logs directory
            LOG_FILE                                # Log file name
        )
    )

    formatter = logging.Formatter(                  # Define log message format
        "%(asctime)s | %(levelname)s | %(message)s" # Timestamp | Level | Message
    )

    file_handler.setFormatter(                      # Attach formatter to file handler
        formatter
    )

    logger.addHandler(                              # Register file handler with logger
        file_handler
    )

    logger.propagate = False                        # Prevent logs from being sent to root logger


# ── Silence Third-Party Library Logs ─────────────────────────────────────────

logging.getLogger(                                  # Access SentenceTransformers logger
    "sentence_transformers"
).setLevel(
    logging.WARNING                                 # Show only warnings and errors
)

logging.getLogger(                                  # Access HuggingFace logger
    "huggingface_hub"
).setLevel(
    logging.WARNING                                 # Hide INFO messages
)

logging.getLogger(                                  # Access Transformers logger
    "transformers"
).setLevel(
    logging.WARNING                                 # Hide INFO messages
)

logging.getLogger(                                  # Access HTTPX logger
    "httpx"
).setLevel(
    logging.WARNING                                 # Hide HTTP request logs
)

logging.getLogger(                                  # Access Qdrant logger
    "qdrant_client"
).setLevel(
    logging.WARNING                                 # Hide Qdrant INFO messages
)