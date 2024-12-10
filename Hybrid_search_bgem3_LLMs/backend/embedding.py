import pandas as pd
from bge_m3 import BGEM3EmbeddingFunction
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import os
import sys
from contextlib import redirect_stdout
from scipy.sparse import csr_matrix

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Global lock for shared resources
global_lock = Lock()
global_model = None  # To hold the global embedding model

# Global lock for progress bar updates
progress_lock = Lock()

def suppress_output(func):
    """
    Decorator to suppress standard output of a function.
    """
    def wrapper(*args, **kwargs):
        with open(os.devnull, 'w') as fnull, redirect_stdout(fnull):
            return func(*args, **kwargs)
    return wrapper

@suppress_output
def load_global_model(device):
    """
    Load the global embedding model once with thread safety.
    """
    global global_model
    with global_lock:
        if global_model is None:
            logging.info("Loading global embedding model...")
            global_model = BGEM3EmbeddingFunction(use_fp16=False, device='cuda')
    return global_model

def generate_embeddings_worker(doc_chunk, id_chunk, device, col):
    """
    Worker function to generate embeddings for a chunk of documents and insert them into Milvus.
    """
    ef = load_global_model(device)  # Use the global model
    logging.info(f" dense_dim: {ef.dim['dense']}")

    try:
        embeddings = ef(doc_chunk)
        dense_embeddings = embeddings["dense"]
        sparse_embeddings = embeddings["sparse"]

        # Convert sparse embeddings to the correct format
        sparse_embeddings = [dict(zip(row.indices, row.data)) for row in sparse_embeddings]

        # Insert directly into Milvus 
        # to do: batch insert and insert can not in the multithreading
        insert_embeddings_into_milvus(col, id_chunk, doc_chunk, dense_embeddings, sparse_embeddings)
    except Exception as e:
        logging.error(f"Error in worker: {e}")

def create_milvus_collection():
    """
    Create a Milvus collection with fields for dense and sparse embeddings.
    """
    logging.info("Connecting to Milvus...")
    connections.connect("default", host="localhost", port="19530")

    # Define Milvus collection fields
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=1024)
    ]

    schema = CollectionSchema(fields, description="Hybrid search collection for short text")
    col_name = 'hybrid_demo'

    # Comment out the following lines to avoid deleting the collection
    # if utility.has_collection(col_name):
    #     logging.info(f"Collection '{col_name}' already exists. Dropping it...")
    #     utility.drop_collection(col_name)

    if not utility.has_collection(col_name):
        col = Collection(col_name, schema, consistency_level="Strong")

        # Create indices
        logging.info("Creating indices for the collection...")
        sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
        col.create_index("sparse_vector", sparse_index)

        dense_index = {"index_type": "FLAT", "metric_type": "IP"}
        col.create_index("dense_vector", dense_index)

        col.load()
        logging.info(f"Milvus collection '{col_name}' created and indexed successfully")
    else:
        col = Collection(col_name)

    return col

def insert_embeddings_into_milvus(col, ids, docs, dense_vectors, sparse_vectors):
    """
    Insert embeddings and related data into the Milvus collection.
    """
    logging.info("Inserting data into Milvus...")

    # Truncate documents to the maximum allowed length
    max_length = 1024
    truncated_docs = [doc[:max_length] for doc in docs]

    # Print sparse vectors for debugging
    # logging.info(f"Sparse vectors to be inserted: {sparse_vectors[:5]}")  # Print first 5 for brevity

    entities = [ids, truncated_docs, sparse_vectors, dense_vectors]  # Ensure all fields are included
    col.insert(entities)
    col.flush()
    logging.info(f"Inserted batch into Milvus collection '{col.name}' successfully")

def generate_and_insert_embeddings(tsv_file, num_threads, batch_size):
    """
    Generate embeddings using multithreading and insert them into Milvus.
    """
    # Load TSV file
    logging.info("Loading TSV file...")
    df = pd.read_csv(tsv_file, sep='\t', header=None, names=['id', 'text'], encoding='utf-8')
    
    # Use only the first 10000 rows
    df = df.head(1000)
    
    docs = df['text'].tolist()
    ids = df['id'].tolist()

    # Check for GPU availability
    device = "cuda"
    logging.info(f"Using device: {device}")

    # Initialize Milvus collection
    col = create_milvus_collection()

    # Generate embeddings in parallel
    logging.info("Generating embeddings in parallel and inserting into Milvus...")
    total_docs = len(docs)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(generate_embeddings_worker, docs[i:i + batch_size], ids[i:i + batch_size], device, col)
            for i in range(0, len(docs), batch_size)
        ]
        for future in futures:
            future.result()  # Wait for each future to complete

    logging.info("All embeddings processed and inserted into Milvus.")

def main(tsv_file, num_threads=12, batch_size=256):
    """
    Main function: generate embeddings, create collection, and insert data.
    """
    generate_and_insert_embeddings(tsv_file, num_threads, batch_size)

if __name__ == "__main__":
    tsv_file = "tmp/collection.tsv"
    main(tsv_file, num_threads=12, batch_size=256)
