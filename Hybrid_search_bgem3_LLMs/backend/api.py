from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import time
import os
from query_expansion import expand_query
app = Flask(__name__)
CORS(app)

COLLECTION_FILE_PATH = 'collection.tsv'

@app.route('/search', methods=['POST','GET'])
def search():
    if request.method == 'GET':
        return jsonify({'message': 'This endpoint only supports POST requests.'}), 405
    if request.content_type != 'application/json':
        return jsonify({'error': 'Content-Type must be application/json'}), 415
 
    data = request.get_json()
    query = data.get('query')
    mode = data.get('mode', '1')  # Default to conjunctive
    if not query:
        return jsonify({'error': 'Query parameter is required'}), 400
    print(f"Received query: {query}, Mode: {mode}")

    # check if the files exist
    files_to_check = ['./query_processor', 'tmp/final_inverted_index.bin']
    for file in files_to_check:
        if not os.path.exists(file):
            print(f"Error: {file} does not exist")
            return jsonify({'error': f'Missing file: {file}'}), 500

     # Step 1: Expand the query
    api_key="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"#change to your own key or set OPENAI_API_KEY environment variable
    if not api_key:
        return jsonify({'error': 'OpenAI API key not found. Please set OPENAI_API_KEY environment variable.'}), 500
    try:
        expanded_query = expand_query(query, api_key)  # 调用扩展函数
        print(f"Show Expanded Query: {expanded_query}")
    except Exception as e:
        return jsonify({'error': f'Error during query expansion: {str(e)}'}), 500
    
    
    # Step 2: Call C++ executable with the expanded query
    try:
        start_time = time.time()
        process = subprocess.Popen(
            ['./query_processor', 'tmp/final_inverted_index.bin', 'tmp/lexicon.txt', COLLECTION_FILE_PATH, expanded_query, mode],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        output, error = process.communicate()
        print(f"Output: {output}")

        if process.returncode != 0 or error:
            return jsonify({'error': f'Query processor error: {error.decode()}'}), 500

        query_terms = expanded_query.split()
        results = parse_cpp_output(output, query_terms)
        processing_time = time.time() - start_time

        return jsonify({
            'results': results,
            'processing_time': processing_time,
            'expanded_query': expanded_query
        })
    except Exception as e:
        return jsonify({'error': f'Error during query processing: {str(e)}'}), 500

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
                        # Shorten the passage based on query terms
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
    """
    Shorten the passage by keeping only `window` words before and after each query term.
    Replace the rest with '...'.
    """
    words = passage.split()
    relevant_indices = set()

    # Find the indices of query terms in the passage
    for idx, word in enumerate(words):
        for query_term in query_terms:
            if query_term.lower() in word.lower():  # Case-insensitive match
                # Add indices for a window around the query term
                relevant_indices.update(range(max(0, idx - window), min(len(words), idx + window + 1)))

    # Create a new passage with only relevant words and '...' for skipped parts
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

    if shortened[-1] != '...':  # Avoid trailing '...' if the last word is relevant
        shortened.append('...')

    return ' '.join(shortened)

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)