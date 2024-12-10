import faiss
import h5py
import numpy as np

# Function to load embeddings from .h5 files
def load_embeddings(h5_file_path):
    with h5py.File(h5_file_path, 'r') as h5_file:
        ids = h5_file['id'][:]  # Load query IDs
        embeddings = h5_file['embedding'][:]  # Load query embeddings
    return ids, embeddings

# Load query embeddings
query_ids, query_embeddings = load_embeddings("msmarco_queries_dev_eval_embeddings.h5")
# Ensure query embeddings are in float32 for compatibility with Faiss
query_embeddings = query_embeddings.astype('float32')

# Load the index from disk
hnsw_index = faiss.read_index("hnsw_index.faiss")
print("Index has been loaded from 'hnsw_index.faiss'")

# Set efSearch parameter for querying (can be adjusted based on your needs)
hnsw_index.hnsw.efSearch = 256  # Higher values can improve accuracy at the cost of speed

# Perform the search for each query to get top-K similar documents
def search_index(index, query_embeddings, top_k=1000):
    distances, indices = index.search(query_embeddings, top_k)  # Retrieve top_k nearest documents
    return distances, indices

# Perform search for top-K documents for each query
top_k = 100
distances, indices = search_index(hnsw_index, query_embeddings, top_k=top_k)

# Load document IDs
doc_ids, _ = load_embeddings("msmarco_passages_embeddings_subset.h5")

# Convert byte strings to strings if necessary
doc_ids = np.array([doc_id.decode('utf-8') if isinstance(doc_id, bytes) else str(doc_id) for doc_id in doc_ids])
query_ids = np.array([query_id.decode('utf-8') if isinstance(query_id, bytes) else str(query_id) for query_id in query_ids])

# Output the results in TREC format
output_file_path = "vector_search_results.txt"

with open(output_file_path, "w") as output_file:
    for i in range(len(query_ids)):
        query_id = query_ids[i]
        retrieved_doc_indices = indices[i]
        retrieved_scores = distances[i]
        retrieved_doc_ids = doc_ids[retrieved_doc_indices]
        for rank, (doc_id, distance) in enumerate(zip(retrieved_doc_ids, retrieved_scores), start=1):
            score = -distance  # As distances are negative inner products
            output_file.write(f"{query_id} Q0 {doc_id} {rank} {score} STANDARD\n")

print(f"Results saved to {output_file_path}")
