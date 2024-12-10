use_bge_m3 = True
use_reranker = True
import numpy as np
from pymilvus import (
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection, connections,
)
import pandas as pd

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# 1. Load the first 10 lines from the TSV file
def load_docs_from_tsv(tsv_file):
    logging.info("Loading TSV file...")
    df = pd.read_csv(tsv_file, sep='\t', header=None, names=['id', 'text'], encoding='utf-8')
    df = df.head(800000)  # Use only the first 800000 rows
    return df['id'].tolist(), df['text'].tolist()

# Load the documents
tsv_file = "tmp/collection.tsv"
ids, docs = load_docs_from_tsv(tsv_file)

# Truncate documents to the maximum allowed length
max_length = 1024
truncated_docs = [doc[:max_length] for doc in docs]

# Print the loaded documents for verification
#print("Loaded documents:")
#for i, doc in enumerate(truncated_docs):
#    print(f"{i+1}: {doc}")

if use_bge_m3:
    from bge_m3 import BGEM3EmbeddingFunction
    ef = BGEM3EmbeddingFunction(use_fp16=False, device="cuda")
    dense_dim = ef.dim["dense"]

# 2. setup Milvus collection and index
connections.connect("default", host="localhost", port="19530")

# Specify the data schema for the new Collection.
fields = [
    # Use id from TSV as primary key
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    # Store the original text to retrieve based on semantically distance
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048),
    # Milvus now supports both sparse and dense vectors, we can store each in
    # a separate field to conduct hybrid search on both vectors.
    FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR,
                dim=dense_dim),
]
schema = CollectionSchema(fields, "")
# define collection name
col_name = 'build_milvus_800k'

# Check if the collection already exists and drop it if it does
if utility.has_collection(col_name):
    logging.info(f"Collection '{col_name}' already exists. Dropping it...")
    utility.drop_collection(col_name)

# Now we can create the new collection with above name and schema.
col = Collection(col_name, schema, consistency_level="Strong")

# We need to create indices for the vector fields. The indices will be loaded
# into memory for efficient search.
sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
col.create_index("sparse_vector", sparse_index)
dense_index = {"index_type": "FLAT", "metric_type": "IP"}
col.create_index("dense_vector", dense_index)
col.load()

# 3. insert text and sparse/dense vector representations into the collection
# Define a smaller batch size for insertion
batch_size = 1000  # Adjust this number based on your memory and performance requirements

# Process and insert data in batches
for i in range(0, len(ids), batch_size):
    batch_ids = ids[i:i + batch_size]
    batch_docs = truncated_docs[i:i + batch_size]

    # Generate embeddings for the current batch
    docs_embeddings = ef(batch_docs)
    batch_sparse = docs_embeddings["sparse"]
    batch_dense = docs_embeddings["dense"]

    # Insert the current batch into Milvus
    entities = [batch_ids, batch_docs, batch_sparse, batch_dense]
    col.insert(entities)
    col.flush()
    logging.info(f"Inserted batch {i // batch_size + 1} into Milvus collection '{col.name}' successfully")

