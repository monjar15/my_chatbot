import os                                              
import pickle                                 # Standard library: serialize Python objects to disk

from langchain_text_splitters import RecursiveCharacterTextSplitter  
from langchain_core.documents import Document                      # LangChain's Document wrapper
from langchain_huggingface import HuggingFaceEmbeddings            
from langchain_qdrant import QdrantVectorStore                     # LangChain adapter for Qdrant vector database
from qdrant_client import QdrantClient                             # Low-level Qdrant Python client
from qdrant_client.models import Distance, VectorParams            # Config objects: cosine distance + vector dimensions


# ── Configuration constants ──────────────────────────────────────────────────

COLLECTION_NAME   = "rag_documents"       # Name of Qdrant collection that stores our vectors
QDRANT_PATH       = "./qdrant_storage"  
CHUNKS_CACHE_PATH = "./chunks_cache.pkl"  # Pickle file that caches split chunks for BM25 retriever re-use
EMBEDDING_MODEL   = "BAAI/bge-m3"         
CHUNK_SIZE        = 768                   
CHUNK_OVERLAP     = 100                   
VECTOR_DIM        = 1024                  


# ── Embedding model loader ───────────────────────────────────────────────────

def get_embedding_model() -> HuggingFaceEmbeddings:

    print(f"[Embedding] Loading embedding model '{EMBEDDING_MODEL}' ...")  

    embeddings = HuggingFaceEmbeddings(                
        model_name=EMBEDDING_MODEL,                    
        model_kwargs={"device": "cpu"},                # Run on CPU; change to "cuda" if you have a GPU
        encode_kwargs={"normalize_embeddings": True}   # normalise vectors
    )

    print(f"[Embedding] ✅ Embedding model loaded.")

    return embeddings                                  


# ── Text splitting ───────────────────────────────────────────────────────────

def split_documents(docs: list[Document]) -> list[Document]:

    splitter = RecursiveCharacterTextSplitter(      
        chunk_size=CHUNK_SIZE,                       
        chunk_overlap=CHUNK_OVERLAP,                 
        length_function=len,                         # Use to measure chunk size in characters
        separators=["\n\n", "\n", ". ", " ", ""]     # Try double newline first, then single, then sentence, then word
    )

    chunks = splitter.split_documents(docs)          

    print(f"[Embedding] Split {len(docs)} document(s) → {len(chunks)} chunk(s) "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")   # Log chunk stats

    return chunks                                    


# ── Build / index vector store ───────────────────────────────────────────────

def build_vectorstore(chunks: list[Document]) -> QdrantVectorStore:
 
    embeddings = get_embedding_model()               

    os.makedirs(QDRANT_PATH, exist_ok=True)          # Create qdrant_storage/ folder if it does not exist yet

    client = QdrantClient(path=QDRANT_PATH)          # Open a Qdrant client

    # ── Recreate collection ──
    existing_names = [                               # Get the names of all collections that already exist
        c.name                                       # .name is the string identifier of each collection
        for c in client.get_collections().collections  # .collections is a list of CollectionDescription objects
    ]

    if COLLECTION_NAME in existing_names:            # If collection already exists from a previous run
        client.delete_collection(COLLECTION_NAME)    # delete it to avoid duplicate vectors
        print(f"[Embedding] Deleted old collection '{COLLECTION_NAME}'")  # Log deletion

    client.create_collection(                        
        collection_name=COLLECTION_NAME,         
        vectors_config=VectorParams(                 # Define the vector space
            size=VECTOR_DIM,                         # Must match the embedding model's output dimension (1024)
            distance=Distance.COSINE                 # Use cosine similarity for nearest-neighbour search
        )
    )
    print(f"[Embedding] Created collection '{COLLECTION_NAME}' (dim={VECTOR_DIM}, distance=COSINE)")  # Confirm creation

    # ── Wrap client with LangChain adapter ──
    vectorstore = QdrantVectorStore(                 
        client=client,                               
        collection_name=COLLECTION_NAME,             # Which collection to read/write
        embedding=embeddings                         # The embedding model used to encode documents and queries
    )

    print(f"[Embedding] Embedding {len(chunks)} chunks and inserting into Qdrant ...")  # Warn user this may take time
    vectorstore.add_documents(chunks)                # Encode every chunk and upsert the vectors into Qdrant
    print(f"[Embedding] ✅ Stored {len(chunks)} chunks in Qdrant at '{QDRANT_PATH}'")   # Confirm storage

    # ── Save chunks for BM25 ──
    with open(CHUNKS_CACHE_PATH, "wb") as f:         # Open (or create) the pickle cache file in binary-write mode
        pickle.dump(chunks, f)                       # Serialise the chunks list to disk
    print(f"[Embedding] ✅ Chunk cache saved to '{CHUNKS_CACHE_PATH}'")  # Confirm cache write

    return vectorstore                              


# ── Load existing vector store ───────────────────────────────────────────────

def load_vectorstore() -> QdrantVectorStore:

    if not os.path.exists(QDRANT_PATH):              # Check that the storage folder is present
        raise FileNotFoundError(                     # Tell the user what to do if it's missing
            f"Qdrant storage not found at '{QDRANT_PATH}'. "
            "Please index your documents first."
        )

    embeddings = get_embedding_model()               # Load the embedding model (needed to encode queries later)

    client = QdrantClient(path=QDRANT_PATH)          

    vectorstore = QdrantVectorStore(                 
        client=client,                               
        collection_name=COLLECTION_NAME,             
        embedding=embeddings                         
    )

    print(f"[Embedding] ✅ Loaded existing vectorstore from '{QDRANT_PATH}'")  # Confirm successful load

    return vectorstore                               


# ── Load cached chunks for BM25 ─────────────────────────────────────────────

def load_chunks_cache() -> list[Document]:

    if not os.path.exists(CHUNKS_CACHE_PATH):        # Check that the cache file exists
        raise FileNotFoundError(                     # Tell the user how to fix it
            f"Chunk cache not found at '{CHUNKS_CACHE_PATH}'. "
            "Please index your documents first."
        )

    with open(CHUNKS_CACHE_PATH, "rb") as f:         # Open the pickle file in binary-read mode
        chunks = pickle.load(f)                      # Deserialise the Python list from disk

    print(f"[Embedding] ✅ Loaded {len(chunks)} chunks from cache '{CHUNKS_CACHE_PATH}'")  # Confirm load

    return chunks                                    