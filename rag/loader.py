import os
from llama_index.core import SimpleDirectoryReader           
from langchain_core.documents import Document                # LangChain's document wrapper (page_content + metadata)


def load_documents(folder_path: str) -> list[Document]:

    print(f"[Loader] Reading documents from: '{folder_path}'")  

    reader = SimpleDirectoryReader(              
        input_dir=folder_path,                   
        recursive=True,                          # Read files inside sub-folders
        exclude_hidden=True                      # Skip hidden files
    )

    llama_docs = reader.load_data()             # Read every file

    langchain_docs: list[Document] = []         

    for doc in llama_docs:                      # Iterate over every document that was loaded
        metadata = dict(doc.metadata or {})     # Copy so we can safely normalise path fields below

        # ── Normalise path fields to the exact os.path.abspath() format used everywhere
        # else in the app: scan_documents_folder(), the recorder, get_doc_filepath(), and
        # the Qdrant deletion filter in remove_documents_by_source(). LlamaIndex's own
        # path resolution can differ slightly (separators, symlinks), which would make
        # deletion/lookup matches silently fail without this step.
        raw_path = metadata.get("file_path") or metadata.get("source")
        if raw_path:
            normalised_path = os.path.abspath(raw_path)
            metadata["file_path"] = normalised_path
            metadata["source"]    = normalised_path

        lc_doc = Document(                      # Wrap the text and meta data
            page_content=doc.text,              # .text holds the raw extracted text content
            metadata=metadata                   # normalised source path, page number, etc. fall back to {} if None
        )
        langchain_docs.append(lc_doc)           # Add the converted document to output list

    print(f"[Loader] ✅ Loaded {len(langchain_docs)} document(s) from '{folder_path}'")  

    return langchain_docs                       
