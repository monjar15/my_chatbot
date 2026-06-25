import re  # Import regular expression module para sa text cleaning

from langchain_core.documents import Document  # LangChain document object

def clean_text(text: str) -> str:
    """
    Clean raw document text before chunking.
    """

    # Remove page numbers: "Page 1,Page 2"
    text = re.sub(
        r"Page\s+\d+",
        "",
        text
    )

    # Convert multiple spaces/tabs/newlines
    # into a single space
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    # Replace multiple blank lines
    # with a single newline
    text = re.sub(
        r"\n{2,}",
        "\n",
        text
    )

    # Remove spaces at beginning/end
    text = text.strip()

    # Return cleaned text
    return text


def clean_documents(
    docs: list[Document]
) -> list[Document]:

    cleaned_docs: list[Document] = []                 # Output container for cleaned documents

    for doc in docs:                                 # Iterate through every loaded document

        cleaned_text = clean_text(                   # Clean the document text
            doc.page_content                         # Original extracted text
        )

        cleaned_doc = Document(                      # Create a new LangChain document
            page_content=cleaned_text,              # Store cleaned text
            metadata=doc.metadata                   # Preserve original metadata
        )

        cleaned_docs.append(                         # Add cleaned document to output list
            cleaned_doc
        )

    print(                                           # Log cleaning summary
        f"[Cleaner] ✅ Cleaned {len(cleaned_docs)} document(s)"
    )

    return cleaned_docs                             # Return cleaned documents