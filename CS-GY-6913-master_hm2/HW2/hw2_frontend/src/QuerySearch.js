import React, { useState } from 'react';
import './QuerySearch.css';

const QuerySearch = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [mode, setMode] = useState('1');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [processingTime, setProcessingTime] = useState(null);
  const [queryWords, setQueryWords] = useState('')

  const handleSearch = async () => {
    setError('');
    setLoading(true);
    setProcessingTime(null);
    try {
      const response = await fetch('http://localhost:5000/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, mode }), // Send both query and mode (conjunctive or disjunctive)
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }

      const data = await response.json();
      setResults(data.results);
      setProcessingTime(data.processing_time.toFixed(4)); // Capture processing time and round to 4 decimals
      setLoading(false);
      setQueryWords(query
        .toLowerCase()
        .split(/\s+/)
        .map((word) => word.replace(/[^\w]/g, ''))
        .filter((word) => word.length > 0))
    } catch (err) {
      setLoading(false);
      setError('Could not connect to the backend. Please try again later.');
      console.error(err);
    }
  };

  // Function to highlight query words in the snippet
  const highlightSnippet = (snippet, query) => {
    if (queryWords.length === 0) {
      return snippet;
    }

    // Escape special characters in query words for regex
    const escapedWords = queryWords.map((word) => word.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&'));

    // Create a regex pattern to match any of the query words, case-insensitive
    const pattern = new RegExp(`\\b(${escapedWords.join('|')})\\b`, 'gi');

    // Replace occurrences in the snippet with <strong> tags
    const highlightedSnippet = snippet.replace(pattern, (match) => `<strong>${match}</strong>`);

    // Render the highlighted snippet safely
    return <span dangerouslySetInnerHTML={{ __html: highlightedSnippet }} />;
  };

  return (
    <div className="query-container">
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Enter search query"
        className="query-input"
      />
      <select onChange={(e) => setMode(e.target.value)} value={mode} className="query-mode">
        <option value="1">Conjunctive</option>
        <option value="2">Disjunctive</option>
      </select>
      <button onClick={handleSearch} className="query-button">
        {loading ? 'Loading' : 'Search'}
      </button>

      {error && <p className="error-message">{error}</p>}

      {processingTime && !loading && (
        <p className="processing-time">Query processed in {processingTime} seconds.</p>
      )}

      {loading && <p>Loading...</p>}

      <div className="results-container">
        {results.map((result) => (
          <div key={result.docID} className="result-item">
            <p>
              <strong>DocID:</strong> {result.docID}
            </p>
            <p>
              <strong>Score:</strong> {result.score}
            </p>
            <p className="snippet">
              {highlightSnippet(result.snippet, query)}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default QuerySearch;
