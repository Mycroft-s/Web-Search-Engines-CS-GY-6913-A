import faiss
import h5py
import numpy as np

# Function to load embeddings from .h5 files
def load_embeddings(h5_file_path):
    with h5py.File(h5_file_path, 'r') as h5_file:
        ids = h5_file['id'][:]  # Load document IDs
        embeddings = h5_file['embedding'][:]  # Load embeddings
    return ids, embeddings

# Load document embeddings
doc_ids, doc_embeddings = load_embeddings("../msmarco_passages_embeddings_subset.h5")
# Ensure document embeddings are in float32 for compatibility with Faiss
doc_embeddings = doc_embeddings.astype('float32')

# Build HNSW index for dense vector search using Faiss
def build_hnsw_index(embeddings, m=8, ef_construction=200):
    dimension = embeddings.shape[1]  # Assuming 384-dimensional embeddings
    index = faiss.IndexHNSWFlat(dimension, m, faiss.METRIC_INNER_PRODUCT)  # dot product similarity
    index.hnsw.efConstruction = ef_construction  # Set construction parameter for indexing
    index.add(embeddings)  # Add document embeddings to the index
    return index

# Create the HNSW index with the specified parameters
hnsw_index = build_hnsw_index(doc_embeddings, m=8, ef_construction=200)

# Save the index
faiss.write_index(hnsw_index, "hnsw_index.faiss")
print("Index has been built and saved to 'hnsw_index.faiss'")
