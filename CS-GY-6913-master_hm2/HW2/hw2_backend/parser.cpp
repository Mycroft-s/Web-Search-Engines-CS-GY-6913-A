#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cctype>

// Define the Posting struct
struct Posting {
    std::string term;
    int docID;
    int freq;
};

// Buffer to hold postings before writing to disk
std::vector<Posting> postingsBuffer;
// const size_t maxBufferSize = 1000000;
const size_t maxBufferSize = 20 * 1024 * 1024; // ~20 MB memory usage

// Global variables
std::vector<std::string> tempFileNames;
std::unordered_map<int, std::string> pageTable;
std::unordered_map<int, int> documentLengths;
std::unordered_map<std::string, int> docFrequencyMap;
std::unordered_map<int, int64_t> passageOffsets; // New map to store passage offsets
int totalDocumentLength = 0;
int totalDocuments = 0;

// Function to check if a string contains only ASCII characters
bool isAscii(const std::string& str) {
    return std::all_of(str.begin(), str.end(), [](unsigned char c) {
        return c >= 0 && c <= 127;
    });
}

// Tokenization function
std::vector<std::string> tokenize(const std::string& text) {
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
        if (isAscii(token)) {
            tokens.push_back(token);
        }
    }

    return tokens;
}


// Function to update the postings buffer with term frequencies
void updatePostingsBuffer(const std::unordered_map<std::string, int>& termFreqMap, int docID) {
    for (const auto& termFreq : termFreqMap) {
        const std::string& term = termFreq.first;
        int freq = termFreq.second;
        postingsBuffer.push_back({term, docID, freq});
        docFrequencyMap[term]++;
    }
}

// Function to write postings buffer to a temporary file
void writePostingsBufferToDisk(int tempFileIndex, const std::string& tempFilePrefix) {
    // Sort postingsBuffer by term and docID
    std::sort(postingsBuffer.begin(), postingsBuffer.end(), [](const Posting& a, const Posting& b) {
        if (a.term == b.term) {
            return a.docID < b.docID; // For the same term, order by docID
        }
        return a.term < b.term; // Order by term
    });

    // Write postingsBuffer to temporary file
    std::string tempFileName = tempFilePrefix + std::to_string(tempFileIndex) + ".txt";
    std::ofstream outFile(tempFileName);
    if (!outFile.is_open()) {
        std::cerr << "Error: Unable to open file for writing: " << tempFileName << std::endl;
        return;
    }

    for (const auto& posting : postingsBuffer) {
        outFile << posting.term << " " << posting.docID << " " << posting.freq << "\n";
    }
    outFile.close();

    tempFileNames.push_back(tempFileName);
    postingsBuffer.clear();
    std::cout << "[INFO] Wrote postings to " << tempFileName << std::endl;
}

// Function to save document frequencies
void saveDocumentFrequencies(const std::string& docFreqFile) {
    std::ofstream outFile(docFreqFile);
    if (!outFile.is_open()) {
        std::cerr << "Error: Unable to open file for writing: " << docFreqFile << std::endl;
        return;
    }

    for (const auto& entry : docFrequencyMap) {
        outFile << entry.first << " " << entry.second << "\n";
    }

    outFile.close();
}

// Function to save document lengths
void saveDocumentLengths(const std::string& docLengthsFile) {
    std::ofstream outFile(docLengthsFile);
    if (!outFile.is_open()) {
        std::cerr << "Error: Unable to open file for writing: " << docLengthsFile << std::endl;
        return;
    }

    for (const auto& entry : documentLengths) {
        outFile << entry.first << " " << entry.second << "\n";
    }

    outFile.close();
}

// Function to save collection statistics
void saveCollectionStats(const std::string& statsFile) {
    std::ofstream outFile(statsFile);
    if (!outFile.is_open()) {
        std::cerr << "Error: Unable to open file for writing: " << statsFile << std::endl;
        return;
    }

    double avgDocumentLength = static_cast<double>(totalDocumentLength) / totalDocuments;
    outFile << totalDocuments << " " << avgDocumentLength << "\n";
    outFile.close();
}

// Function to save the page table
void savePageTable(const std::string& pageTableFile) {
    std::ofstream outFile(pageTableFile);
    if (!outFile.is_open()) {
        std::cerr << "Error: Unable to open file for writing: " << pageTableFile << std::endl;
        return;
    }

    for (const auto& entry : pageTable) {
        int docID = entry.first;
        const std::string& docName = entry.second;
        outFile << docID << " " << docName << "\n";
    }

    outFile.close();
}

// Function to save passage offsets
void savePassageOffsets(const std::string& passageOffsetsFile) {
    std::ofstream outFile(passageOffsetsFile);
    if (!outFile.is_open()) {
        std::cerr << "Error: Unable to open file for writing: " << passageOffsetsFile << std::endl;
        return;
    }

    for (const auto& entry : passageOffsets) {
        int docID = entry.first;
        int64_t offset = entry.second;
        outFile << docID << " " << offset << "\n";
    }

    outFile.close();
}

// Main parsing function
void parseDocuments(const std::string& filePath, const std::string& tempFilePrefix) {
    std::ifstream file(filePath);
    std::string line;
    int docID = 0;
    int tempFileIndex = 0;

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filePath << std::endl;
        return;
    }

    while (true) {
        int64_t offset = file.tellg(); // Record the offset before reading the line
        if (!std::getline(file, line)) {
            break;
        }
        passageOffsets[docID] = offset; // Store the offset for this docID
        totalDocuments++;

        std::istringstream ss(line);
        std::string passageID, passageText;
        std::getline(ss, passageID, '\t');
        std::getline(ss, passageText, '\t'); 

        pageTable[docID] = passageID;

        // Tokenize and calculate term frequencies
        std::vector<std::string> tokens = tokenize(passageText);
        int docLength = tokens.size();
        documentLengths[docID] = docLength;
        totalDocumentLength += docLength;

        std::unordered_map<std::string, int> termFreqMap;
        for (const std::string& token : tokens) {
            termFreqMap[token]++;
        }

        // Update postings buffer
        updatePostingsBuffer(termFreqMap, docID);

        // Write to disk if buffer is full
        if (postingsBuffer.size() >= maxBufferSize) {
            tempFileIndex++;
            writePostingsBufferToDisk(tempFileIndex, tempFilePrefix);
        }
        docID++;
    }

    // Write any remaining postings to disk
    if (!postingsBuffer.empty()) {
        tempFileIndex++;
        writePostingsBufferToDisk(tempFileIndex, tempFilePrefix);
    }

    file.close();

    // Save document frequencies, document lengths, collection stats, page table, and passage offsets
    saveDocumentFrequencies("tmp/doc_frequencies.txt");
    saveDocumentLengths("tmp/document_lengths.txt");
    saveCollectionStats("tmp/collection_stats.txt");
    savePageTable("tmp/page_table.txt");
    savePassageOffsets("tmp/passage_offsets.txt"); // Save passage offsets
    std::cout << "[INFO] Parsing completed." << std::endl;
}