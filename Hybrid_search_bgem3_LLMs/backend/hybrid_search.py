from pymilvus import (
    Collection, AnnSearchRequest, RRFRanker, connections
)
from bge_m3 import BGEM3EmbeddingFunction  # If using BGE models
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def hybrid_search_with_rerank(query_text, collection_name='build_milvus_800k', use_reranker=True, top_k=5):
    """
    Perform hybrid search on a Milvus collection and optionally rerank the results.

    Args:
    - query_text (str): The query text to search for.
    - collection_name (str): The name of the Milvus collection.
    - use_reranker (bool): Whether to use the BGE CrossEncoder for reranking.
    - top_k (int): The number of top results to retrieve.
    """
    # Connect to Milvus
    logging.info("Connecting to Milvus...")
    connections.connect("default", host="localhost", port="19530")

    # Load the specified collection
    logging.info(f"Loading collection '{collection_name}'...")
    col = Collection(collection_name)
    col.load()

    # Generate query embeddings using the BGE model
    logging.info("Loading BGE model and generating query embeddings...")
    ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
    query_embeddings = ef([query_text])

    # Prepare search parameters
    logging.info("Preparing search parameters...")
    sparse_search_params = {"metric_type": "IP"}
    sparse_req = AnnSearchRequest(query_embeddings["sparse"], "sparse_vector", sparse_search_params, limit=top_k)
    dense_search_params = {"metric_type": "IP"}
    dense_req = AnnSearchRequest(query_embeddings["dense"], "dense_vector", dense_search_params, limit=top_k)

    # Perform hybrid search
    logging.info(f"Performing hybrid search on collection '{collection_name}'...")
    res = col.hybrid_search([sparse_req, dense_req], rerank=RRFRanker(), limit=top_k, output_fields=['text'])
    res = res[0]  # Currently, Milvus supports only single-query hybrid search

    # Optionally rerank the results
    if use_reranker:
        logging.info("Using BGE model to rerank search results...")
        result_texts = [hit.fields["text"] for hit in res]
        from bge_reranker import BGERerankFunction
        reranker = BGERerankFunction(device='cpu')
        results = reranker(query_text, result_texts, top_k=top_k)
        logging.info("Reranked results:")
        for hit in results:
            print(f'Text: {hit.text}, Score: {hit.score}')
        return results
    else:
        logging.info("Original search results:")
        for hit in res:
            print(f'Text: {hit.fields["text"]}, Distance: {hit.distance}')
        return res # return the original results


if __name__ == "__main__":
    # Example query
    query = "What is manhattan project?"
    hybrid_search_with_rerank(query, collection_name='build_milvus_800k', use_reranker=True, top_k=10)
