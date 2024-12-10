import numpy as np
from api import retrieve_documents  # 假设这个函数在api.py中实现

# 示例测试集
test_queries = [
    {
        "query": "example query 1",
        "relevant_docs": [1, 2, 3]  # 相关文档的ID
    },
    {
        "query": "example query 2",
        "relevant_docs": [4, 5]
    },
    # 更多查询...
]

def calculate_recall_at_k(retrieved_docs, relevant_docs, k=10):
    retrieved_set = set(retrieved_docs[:k])
    relevant_set = set(relevant_docs)
    return len(retrieved_set & relevant_set) / len(relevant_set)

def calculate_dcg(retrieved_docs, relevant_docs, k=10):
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_docs[:k]):
        if doc_id in relevant_docs:
            dcg += 1 / np.log2(i + 2)  # i+2 because log2(1) = 0
    return dcg

def calculate_ndcg(retrieved_docs, relevant_docs, k=10):
    dcg = calculate_dcg(retrieved_docs, relevant_docs, k)
    ideal_dcg = calculate_dcg(sorted(relevant_docs, reverse=True), relevant_docs, k)
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def calculate_average_precision(retrieved_docs, relevant_docs):
    relevant_set = set(relevant_docs)
    num_relevant = 0
    sum_precision = 0.0

    for i, doc_id in enumerate(retrieved_docs):
        if doc_id in relevant_set:
            num_relevant += 1
            sum_precision += num_relevant / (i + 1)

    return sum_precision / len(relevant_set) if relevant_set else 0.0

def calculate_map(test_queries):
    average_precisions = []
    for query_data in test_queries:
        query = query_data["query"]
        relevant_docs = query_data["relevant_docs"]
        retrieved_docs = retrieve_documents(query)  # 假设这是一个返回文档ID列表的函数
        average_precisions.append(calculate_average_precision(retrieved_docs, relevant_docs))
    return np.mean(average_precisions)

def evaluate_search_system():
    # 计算Recall@10, nDCG, MAP
    for query_data in test_queries:
        query = query_data["query"]
        relevant_docs = query_data["relevant_docs"]
        retrieved_docs = retrieve_documents(query)

        recall_at_10 = calculate_recall_at_k(retrieved_docs, relevant_docs, k=10)
        ndcg = calculate_ndcg(retrieved_docs, relevant_docs, k=10)
        print(f"Query: {query}")
        print(f"Recall@10: {recall_at_10:.4f}")
        print(f"nDCG: {ndcg:.4f}")

    map_score = calculate_map(test_queries)
    print(f"MAP: {map_score:.4f}")

if __name__ == "__main__":
    evaluate_search_system()
