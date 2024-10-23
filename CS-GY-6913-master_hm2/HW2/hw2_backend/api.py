from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import time

app = Flask(__name__)
CORS(app)

COLLECTION_FILE_PATH = 'collection.tsv'

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data.get('query')
    mode = data.get('mode', '1')  # Default to conjunctive

    print(f"Received query: {query}, Mode: {mode}")

    start_time = time.time()

    process = subprocess.Popen(['./query_processor', 'tmp/final_inverted_index.bin', 'tmp/lexicon.txt', COLLECTION_FILE_PATH, query, mode],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()

    if error:
        print(f"Error: {error.decode()}")
        return jsonify({'error': 'Failed to process query'}), 500

    query_terms = query.split()
    results = parse_cpp_output(output, query_terms)
    processing_time = time.time() - start_time

    return jsonify({
        'results': results,
        'processing_time': processing_time
    })

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
    app.run(debug=True)
