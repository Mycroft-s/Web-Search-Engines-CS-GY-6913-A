from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import os
from hybrid_search import hybrid_search_with_rerank
from query_expansion import expand_query

app = Flask(__name__)
CORS(app)

@app.route('/hybrid_search', methods=['POST'])
def hybrid_search():
    if request.content_type != 'application/json':
        return jsonify({'error': 'Content-Type must be application/json'}), 415

    data = request.get_json()
    query = data.get('query')
    top_k = data.get('top_k', 10)  # Default to top 10 results if not specified
    if not query:
        return jsonify({'error': 'Query parameter is required'}), 400

    print(f"Received query: {query}, Top K: {top_k}")

    # Expand the query
    api_key="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"#change to your own key or set OPENAI_API_KEY environment variable
    if not api_key:
        return jsonify({'error': 'OpenAI API key not found. Please set OPENAI_API_KEY environment variable.'}), 500
    try:
        expanded_query = expand_query(query, api_key)
        print(f"Expanded Query: {expanded_query}")
    except Exception as e:
        return jsonify({'error': f'Error during query expansion: {str(e)}'}), 500

    # Execute the hybrid search with rerank
    try:
        start_time = time.time()
        results = hybrid_search_with_rerank(expanded_query, collection_name='build_milvus', use_reranker=True, top_k=top_k)
        processing_time = time.time() - start_time

        # Extract text and score from results
        formatted_results = [{'text': hit.text, 'score': hit.score} for hit in results]
        query_terms = expanded_query.split()
        results = parse_cpp_output(formatted_results, query_terms)

        return jsonify({
            'results': results,
            'processing_time': processing_time,
            'expanded_query': expanded_query
        })
    except Exception as e:
        return jsonify({'error': f'Error during hybrid search: {str(e)}'}), 500

def parse_cpp_output(output, query_terms):
    results = []
    lines = output.decode().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("DocID:"):
            try:
                docID_part, score_part = line.split(', ')
                docID = int(docID_part.split(': ')[1])
                score = float(score_part.split(': ')[1])
                i += 1
                if i < len(lines):
                    passage_line = lines[i]
                    if passage_line.startswith("Passage:"):
                        passage = passage_line[len("Passage: "):]
                        passage = shorten_passage(passage, query_terms)
                    else:
                        passage = ""
                else:
                    passage = ""
                results.append({'docID': docID, 'score': score, 'snippet': passage})
            except ValueError as e:
                print(f"Error parsing line: {line} -> {e}")
        i += 1
    return results

def shorten_passage(passage, query_terms, window=5):
    words = passage.split()
    relevant_indices = set()

    for idx, word in enumerate(words):
        for query_term in query_terms:
            if query_term.lower() in word.lower():
                relevant_indices.update(range(max(0, idx - window), min(len(words), idx + window + 1)))

    shortened = []
    in_relevant_section = False

    for idx, word in enumerate(words):
        if idx in relevant_indices:
            if not in_relevant_section and shortened:
                shortened.append('...')
            shortened.append(word)
            in_relevant_section = True
        else:
            in_relevant_section = False

    if shortened[-1] != '...':
        shortened.append('...')

    return ' '.join(shortened)

if __name__ == '__main__':
    print("Starting Flask server for hybrid search...")
    app.run(host='0.0.0.0', port=5000, debug=True)
