import h5py
import numpy as np

# Paths to the necessary files
bm25_results_file = "../Part1-BM25/bm25_results.txt"
query_embeddings_file = "../msmarco_queries_dev_eval_embeddings.h5"
passage_embeddings_file = "../msmarco_passages_embeddings_subset.h5"
output_file_path = "reranked_results.txt"

# Step 1: Read BM25 results and collect candidate passages
print("Reading BM25 results...")
queries_dict = {}  # queryID -> list of (passageID, BM25score)

with open(bm25_results_file, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) != 6:
            continue  # Skip invalid lines
        queryID, _, passageID, rank, score, _ = parts
        score = float(score)
        if queryID not in queries_dict:
            queries_dict[queryID] = []
        queries_dict[queryID].append((passageID, score))

# Collect sets of queryIDs and passageIDs
queryIDs = set(queries_dict.keys())
passageIDs = set()
for candidate_list in queries_dict.values():
    for passageID, _ in candidate_list:
        passageIDs.add(passageID)

print(f"Total unique queries: {len(queryIDs)}")
print(f"Total unique passages in BM25 results: {len(passageIDs)}")

# Step 2: Build passageID to index mapping
print("Building passage ID to index mapping...")
with h5py.File(passage_embeddings_file, 'r') as h5_file:
    ids = h5_file['id'][:]
    embeddings_dataset = h5_file['embedding']
    # Convert ids to strings
    ids = [id_.decode('utf-8') if isinstance(id_, bytes) else str(id_) for id_ in ids]
    passage_id_to_idx = {id_: idx for idx, id_ in enumerate(ids)}

# Step 3: Build queryID to index mapping
print("Building query ID to index mapping...")
with h5py.File(query_embeddings_file, 'r') as h5_file:
    query_ids = h5_file['id'][:]
    query_embeddings_dataset = h5_file['embedding']
    query_ids = [id_.decode('utf-8') if isinstance(id_, bytes) else str(id_) for id_ in query_ids]
    query_id_to_idx = {id_: idx for idx, id_ in enumerate(query_ids)}

# Step 4: Re-rank and output the results
print("Re-ranking and writing results to file...")
with open(output_file_path, 'w') as output_file:
    with h5py.File(passage_embeddings_file, 'r') as passage_h5:
        embeddings_dataset = passage_h5['embedding']
        with h5py.File(query_embeddings_file, 'r') as query_h5:
            query_embeddings_dataset = query_h5['embedding']
            for queryID in queries_dict:
                if queryID not in query_id_to_idx:
                    print(f"Skipping query {queryID}, embedding not found")
                    continue
                query_idx = query_id_to_idx[queryID]
                query_embedding = query_embeddings_dataset[query_idx]
                candidate_list = queries_dict[queryID]
                scores = []
                for passageID, bm25_score in candidate_list:
                    if passageID in passage_id_to_idx:
                        passage_idx = passage_id_to_idx[passageID]
                        passage_embedding = embeddings_dataset[passage_idx]
                        similarity = np.dot(query_embedding, passage_embedding)
                        scores.append((passageID, similarity))
                    else:
                        print(f"Passage embedding not found for passageID {passageID}")
                        # Assign a very low similarity score for missing embeddings
                        scores.append((passageID, -float('inf')))

                # Sort candidate passages by similarity scores (descending)
                scores.sort(key=lambda x: x[1], reverse=True)

                # Output in TREC format
                for rank, (passageID, similarity_score) in enumerate(scores, start=1):
                    output_file.write(f"{queryID} Q0 {passageID} {rank} {similarity_score} STANDARD\n")

print(f"Re-ranked results saved to {output_file_path}")
