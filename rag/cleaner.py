import re
from langchain_core.documents import Document
from rag.logger import logger  # Shared logger instance — consistent with the rest of the app

# ══════════════════════════════════════════════════════════════════════════
# COMPILED REGEX PATTERNS
# Compiled once at module load instead of on every clean_text() call —
# faster when cleaning hundreds/thousands of chunks.
# ══════════════════════════════════════════════════════════════════════════

_PAGE_NUMBER_PATTERN = re.compile(r"Page\s+\d+", re.IGNORECASE)
_MULTIPLE_WHITESPACE_PATTERN = re.compile(r"[ \t]+")
_MULTIPLE_BLANK_LINES_PATTERN = re.compile(r"\n{2,}")
_MULTIPLE_NEWLINE_SPACES_PATTERN = re.compile(r"\n[ \t]+")  # trailing spaces after a newline


def clean_text(text: str) -> str:

    if not text:
        return ""  # Guard against None/empty text from faulty extractors

    # Remove page-number artifacts (e.g. "Page 1", "page 12")
    text = _PAGE_NUMBER_PATTERN.sub("", text)

    # Collapse repeated spaces/tabs (but NOT newlines — preserves paragraphs)
    text = _MULTIPLE_WHITESPACE_PATTERN.sub(" ", text)

    # Strip stray whitespace that lands right after a line break
    text = _MULTIPLE_NEWLINE_SPACES_PATTERN.sub("\n", text)

    # Collapse 2+ consecutive blank lines into a single newline
    text = _MULTIPLE_BLANK_LINES_PATTERN.sub("\n", text)

    # Trim leading/trailing whitespace
    text = text.strip()

    return text


def clean_documents(docs: list[Document]) -> list[Document]:

    cleaned_docs: list[Document] = []
    skipped_count = 0

    for doc in docs:
        cleaned_text = clean_text(doc.page_content)

        if not cleaned_text:
            skipped_count += 1
            continue  # Don't keep empty documents — they add no value to retrieval

        cleaned_docs.append(
            Document(
                page_content=cleaned_text,
                metadata=doc.metadata,
            )
        )

    logger.info(
        f"[Cleaner] Cleaned {len(cleaned_docs)} document(s)"
        + (f" · skipped {skipped_count} empty" if skipped_count else "")
    )

    return cleaned_docs