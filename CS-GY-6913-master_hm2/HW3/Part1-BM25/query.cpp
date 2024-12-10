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
#include <chrono> // Included for time measurement

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

std::unordered_map<int, std::string> pageTable;

void loadPageTable(const std::string& pageTableFile) {
    std::ifstream inFile(pageTableFile);
    if (!inFile.is_open()) {
        std::cerr << "Error opening page table file!" << std::endl;
        return;
    }

    int docID;
    std::string passageID;
    while (inFile >> docID >> passageID) {
        pageTable[docID] = passageID;
    }

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
    // Start time measurement
    auto startTime = std::chrono::high_resolution_clock::now();

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
            int termFreq = static_cast<int>(invLists[i]->getScore());
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

    // End time measurement
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;

    // Collect and sort the top-k results
    std::vector<DocScore> sortedResults;
    while (!topK.empty()) {
        sortedResults.push_back(topK.top());
        topK.pop();
    }

    std::sort(sortedResults.begin(), sortedResults.end(), [](const DocScore& a, const DocScore& b) {
        return a.score > b.score;
    });

    // Output the time taken
    std::cout << "Query processed in " << elapsed.count() << " seconds." << std::endl;

    // Output top-k results
    std::cout << "Top " << k << " documents:" << std::endl;
    for (const auto& result : sortedResults) {
        std::cout << "DocID: " << result.docID << ", Score: " << result.score << std::endl;
    }
}

std::vector<DocScore> processDisjunctiveQuery(const std::vector<std::string>& terms, IndexAPI& indexAPI, int k) {
    // Open all inverted lists
    std::vector<InvertedList*> invLists;
    std::vector<std::string> validTerms;
    for (const std::string& term : terms) {
        InvertedList* invList = indexAPI.openList(term);
        if (invList != nullptr) {
            invLists.push_back(invList);
            validTerms.push_back(term);
        }
    }

    if (invLists.empty()) {
        // No matching documents
        return {};
    }

    // Initialize pointers for all lists
    std::vector<int> currentDocIDs(invLists.size(), 0);
    for (size_t i = 0; i < invLists.size(); ++i) {
        currentDocIDs[i] = invLists[i]->nextGEQ(0);
    }

    std::unordered_map<int, double> docScores;

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
                int termFreq = static_cast<int>(invLists[i]->getScore());
                int docFrequency = indexAPI.lexicon[validTerms[i]].docFrequency;
                int documentLength = documentLengths[minDocID];
                score += computeBM25(termFreq, docFrequency, documentLength);

                // Advance the list
                currentDocIDs[i] = invLists[i]->nextGEQ(minDocID + 1);
            }
        }

        docScores[minDocID] = score;
    }

    // Close all inverted lists
    for (auto list : invLists) {
        indexAPI.closeList(list);
    }

    // Collect and sort the top-k results
    std::vector<DocScore> sortedResults;
    for (const auto& pair : docScores) {
        sortedResults.push_back({ pair.first, pair.second });
    }

    std::sort(sortedResults.begin(), sortedResults.end(), [](const DocScore& a, const DocScore& b) {
        return a.score > b.score;
    });

    if (sortedResults.size() > static_cast<size_t>(k)) {
        sortedResults.resize(k);
    }

    return sortedResults;
}


std::vector<std::pair<int, std::string>> loadQueries(const std::string& queryFilePath) {
    std::vector<std::pair<int, std::string>> queries;
    std::ifstream inFile(queryFilePath);
    if (!inFile.is_open()) {
        std::cerr << "Error opening query file: " << queryFilePath << std::endl;
        return queries;
    }

    std::string line;
    while (std::getline(inFile, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        int queryID;
        std::string queryText;
        if (!(iss >> queryID)) {
            std::cerr << "Error parsing query ID in line: " << line << std::endl;
            continue;
        }
        std::getline(iss, queryText);
        // Remove leading whitespace from queryText
        queryText.erase(0, queryText.find_first_not_of(" \t"));
        queries.emplace_back(queryID, queryText);
    }

    inFile.close();
    return queries;
}

void startQueryProcessor(const std::string& indexFilePath, const std::string& lexiconFilePath, const std::string& queryFilePath, const std::string& outputFilePath) {
    loadDocumentLengths("tmp/document_lengths.txt");
    loadCollectionStats("tmp/collection_stats.txt");
    loadPageTable("tmp/page_table.txt");

    IndexAPI indexAPI(indexFilePath, lexiconFilePath);

    // Load queries from file
    std::vector<std::pair<int, std::string>> queries = loadQueries(queryFilePath);
    if (queries.empty()) {
        std::cerr << "No queries found in the file: " << queryFilePath << std::endl;
        return;
    }

    // Open output file
    std::ofstream outFile(outputFilePath);
    if (!outFile.is_open()) {
        std::cerr << "Error opening output file: " << outputFilePath << std::endl;
        return;
    }

    // Process each query
    const int k = 1000; // Number of top documents to return
    for (const auto& queryPair : queries) {
        int queryID = queryPair.first;
        std::string queryText = queryPair.second;

        std::vector<std::string> terms = tokenizeQuery(queryText);
        if (terms.empty()) {
            std::cout << "No terms found in query ID " << queryID << "." << std::endl;
            continue;
        }

        // Process disjunctive query and collect results
        std::vector<DocScore> results = processDisjunctiveQuery(terms, indexAPI, k);

        // Write results to output file
        int rank = 1;
        for (const auto& result : results) {
            std::string passageID = pageTable[result.docID];  // Use passageID instead of docID
            outFile << queryID << " Q0 " << passageID << " " << rank << " " << result.score << " STANDARD" << std::endl;
            rank++;
        }
    }

    outFile.close();
}
