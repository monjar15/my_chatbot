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
        lc_doc = Document(                      # Wrap the text and meta data
            page_content=doc.text,              # .text holds the raw extracted text content
            metadata=doc.metadata or {}         # .metadata holds source path, page number, etc.; fall back to {} if None
        )
        langchain_docs.append(lc_doc)           # Add the converted document to output list

    print(f"[Loader] ✅ Loaded {len(langchain_docs)} document(s) from '{folder_path}'")  

    return langchain_docs                       
