#include "index_api.h"
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <queue>
#include <cstdint>

// Data structures to hold loaded data
std::unordered_map<int, int> documentLengths;
int totalDocuments = 0;
double avgDocumentLength = 0.0;

// Define a struct to store document and score for top-k results
struct DocScore {
    int docID;
    double score;
    bool operator<(const DocScore& other) const {
        return score < other.score;  // Min-heap for lowest scores
    }
};

// Function to load document lengths
void loadDocumentLengths(const std::string& docLengthsFile) {
    std::ifstream inFile(docLengthsFile);
    if (!inFile.is_open()) {
        std::cerr << "Error opening document lengths file!" << std::endl;
        return;
    }

    int docID, length;
    while (inFile >> docID >> length) {
        documentLengths[docID] = length;
    }

    inFile.close();
}

// Function to load collection statistics
void loadCollectionStats(const std::string& statsFile) {
    std::ifstream inFile(statsFile);
    if (!inFile.is_open()) {
        std::cerr << "Error opening collection stats file!" << std::endl;
        return;
    }

    inFile >> totalDocuments >> avgDocumentLength;

    inFile.close();
}

// Tokenization function
std::vector<std::string> tokenizeQuery(const std::string& text) {
    std::vector<std::string> tokens;
    std::string processedText = text;

    // Remove punctuation and convert to lowercase
    std::transform(processedText.begin(), processedText.end(), processedText.begin(),
        [](char c) -> char {
            if (std::ispunct(static_cast<unsigned char>(c))) {
                return ' ';
            } else {
                return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            }
        });

    std::istringstream tokenStream(processedText);
    std::string token;
    while (tokenStream >> token) {
        tokens.push_back(token);
    }

    return tokens;
}

// BM25 computation
double computeBM25(int termFrequency, int docFrequency, int documentLength) {
    double k1 = 1.5;
    double b = 0.75;
    double idf = std::log((static_cast<double>(totalDocuments) - static_cast<double>(docFrequency) + 0.5) /
                           (static_cast<double>(docFrequency) + 0.5) + 1.0);

    double tfComponent = (static_cast<double>(termFrequency) * (k1 + 1.0)) /
                         (static_cast<double>(termFrequency) + k1 * (1.0 - b + b * (static_cast<double>(documentLength) / avgDocumentLength)));
    return idf * tfComponent;
}

// Conjunctive Query Processing
void processConjunctiveQuery(const std::vector<std::string>& terms, IndexAPI& indexAPI, int k) {
    // Open all inverted lists
    std::vector<InvertedList*> invLists;
    for (const std::string& term : terms) {
        InvertedList* invList = indexAPI.openList(term);
        if (invList != nullptr) {
            invLists.push_back(invList);
        } else {
            // If any term is not found, no documents can satisfy the conjunctive query
            std::cout << "No matching documents found." << std::endl;
            // Close any opened lists
            for (auto list : invLists) {
                indexAPI.closeList(list);
            }
            return;
        }
    }

    // Initialize pointers for all lists
    std::vector<int> currentDocIDs(invLists.size(), 0);
    for (size_t i = 0; i < invLists.size(); ++i) {
        currentDocIDs[i] = invLists[i]->nextGEQ(0);
        if (currentDocIDs[i] == INT32_MAX) {
            // No documents in one of the lists
            std::cout << "No matching documents found." << std::endl;
            for (auto list : invLists) {
                indexAPI.closeList(list);
            }
            return;
        }
    }

    std::priority_queue<DocScore> topK;

    int did = 0;
    while (did <= INT32_MAX) {
        did = currentDocIDs[0];
        // Find the docID in intersection
        for (size_t i = 1; i < invLists.size(); ++i) {
            int d = currentDocIDs[i];
            while (d < did) {
                d = invLists[i]->nextGEQ(did);
                currentDocIDs[i] = d;
                if (d == INT32_MAX) break;
            }
            if (d > did) {
                did = d;
                i = -1; // Restart the loop
            }
        }
        if (did == INT32_MAX) {
            break;
        }

        // Compute BM25 score for the matched document
        double score = 0.0;
        for (size_t i = 0; i < invLists.size(); ++i) {
            int termFreq = static_cast<int>(invLists[i]->getScore()); // Assuming getScore returns term frequency
            int docFrequency = indexAPI.lexicon[terms[i]].docFrequency;
            int documentLength = documentLengths[did];
            score += computeBM25(termFreq, docFrequency, documentLength);
            // Advance to next posting
            currentDocIDs[i] = invLists[i]->nextGEQ(did + 1);
        }

        // Insert into topK heap
        if (topK.size() < static_cast<size_t>(k)) {
            topK.push({ did, score });
        } else if (score > topK.top().score) {
            topK.pop();
            topK.push({ did, score });
        }
    }

    // Close all inverted lists
    for (auto list : invLists) {
        indexAPI.closeList(list);
    }

    // Collect and sort the top-k results
    std::vector<DocScore> sortedResults;
    while (!topK.empty()) {
        sortedResults.push_back(topK.top());
        topK.pop();
    }

    std::sort(sortedResults.begin(), sortedResults.end(), [](const DocScore& a, const DocScore& b) {
        return a.score > b.score;
    });

    // Output top-k results
    std::cout << "Top " << k << " documents:" << std::endl;
    for (const auto& result : sortedResults) {
        std::cout << "DocID: " << result.docID << ", Score: " << result.score << std::endl;
    }
}

// Disjunctive Query Processing
void processDisjunctiveQuery(const std::vector<std::string>& terms, IndexAPI& indexAPI, int k) {
    // Open all inverted lists
    std::vector<InvertedList*> invLists;
    for (const std::string& term : terms) {
        InvertedList* invList = indexAPI.openList(term);
        if (invList != nullptr) {
            invLists.push_back(invList);
        }
    }

    if (invLists.empty()) {
        std::cout << "No matching documents found." << std::endl;
        return;
    }

    // Initialize pointers for all lists
    std::vector<int> currentDocIDs(invLists.size(), 0);
    for (size_t i = 0; i < invLists.size(); ++i) {
        currentDocIDs[i] = invLists[i]->nextGEQ(0);
    }

    std::priority_queue<DocScore> topK;

    while (true) {
        // Find the minimum docID among current pointers
        int minDocID = INT32_MAX;
        for (size_t i = 0; i < currentDocIDs.size(); ++i) {
            if (currentDocIDs[i] < minDocID) {
                minDocID = currentDocIDs[i];
            }
        }

        if (minDocID == INT32_MAX) {
            break; // All lists exhausted
        }

        // Accumulate scores from all lists that have minDocID
        double score = 0.0;
        for (size_t i = 0; i < invLists.size(); ++i) {
            if (currentDocIDs[i] == minDocID) {
                int termFreq = static_cast<int>(invLists[i]->getScore()); // Assuming getScore returns term frequency
                int docFrequency = indexAPI.lexicon[terms[i]].docFrequency;
                int documentLength = documentLengths[minDocID];
                score += computeBM25(termFreq, docFrequency, documentLength);

                // Advance the list
                currentDocIDs[i] = invLists[i]->nextGEQ(minDocID + 1);
            }
        }

        // Insert into topK heap
        if (topK.size() < static_cast<size_t>(k)) {
            topK.push({ minDocID, score });
        } else if (score > topK.top().score) {
            topK.pop();
            topK.push({ minDocID, score });
        }
    }

    // Close all inverted lists
    for (auto list : invLists) {
        indexAPI.closeList(list);
    }

    // Collect and sort the top-k results
    std::vector<DocScore> sortedResults;
    while (!topK.empty()) {
        sortedResults.push_back(topK.top());
        topK.pop();
    }

    std::sort(sortedResults.begin(), sortedResults.end(), [](const DocScore& a, const DocScore& b) {
        return a.score > b.score;
    });

    // Output top-k results
    std::cout << "Top " << k << " documents:" << std::endl;
    for (const auto& result : sortedResults) {
        std::cout << "DocID: " << result.docID << ", Score: " << result.score << std::endl;
    }
}

void startQueryProcessor(const std::string& indexFilePath, const std::string& lexiconFilePath) {
    loadDocumentLengths("tmp/document_lengths.txt");
    loadCollectionStats("tmp/collection_stats.txt");

    IndexAPI indexAPI(indexFilePath, lexiconFilePath);

    std::string query;
    while (true) {
        std::cout << "Enter query (or 'exit' to quit): ";
        std::getline(std::cin, query);
        if (query == "exit") break;

        std::cout << "Select mode (1: conjunctive, 2: disjunctive): ";
        std::string mode;
        std::getline(std::cin, mode);
        bool conjunctive = (mode == "1");

        std::vector<std::string> terms = tokenizeQuery(query);
        if (terms.empty()) {
            std::cout << "No terms found in query." << std::endl;
            continue;
        }

        const int k = 10; // Number of top documents to return

        if (conjunctive) {
            processConjunctiveQuery(terms, indexAPI, k);
        } else {
            processDisjunctiveQuery(terms, indexAPI, k);
        }
    }
}
