#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <queue>
#include <functional>
#include <unordered_map>
#include <cstdint>

// Define the Posting struct
struct Posting {
    std::string term;
    int docID;
    int freq;
    int fileIndex; // Index of the file this posting came from

    // Constructor
    Posting(const std::string& t, int d, int f, int idx) : term(t), docID(d), freq(f), fileIndex(idx) {}
};

// Comparator for the priority queue (min-heap)
struct PostingComparator {
    bool operator()(const Posting& a, const Posting& b) {
        if (a.term == b.term) {
            return a.docID > b.docID; // For the same term, order by docID
        }
        return a.term > b.term; // Order by term
    }
};

// Variable-byte encoding function
void varByteEncode(int number, std::vector<std::uint8_t>& encodedBytes) {
    while (true) {
        std::uint8_t byte = number & 0x7F;
        number >>= 7;
        if (number == 0) {
            byte |= 0x80; // Set the continuation bit
            encodedBytes.push_back(byte);
            break;
        } else {
            encodedBytes.push_back(byte);
        }
    }
}

// Function to perform I/O-efficient multi-way merge and generate the final inverted index
void mergeInvertedIndexes(const std::vector<std::string>& indexFiles, const std::string& outputIndexFile, const std::string& outputLexiconFile) {
    // Open all temporary posting files
    int numFiles = indexFiles.size();
    std::vector<std::ifstream> inputFiles(numFiles);
    for (int i = 0; i < numFiles; ++i) {
        inputFiles[i].open(indexFiles[i]);
        if (!inputFiles[i].is_open()) {
            std::cerr << "Error opening file: " << indexFiles[i] << std::endl;
            return;
        }
    }

    // Initialize priority queue (min-heap)
    std::priority_queue<Posting, std::vector<Posting>, PostingComparator> pq;

    // Read the first posting from each file and add it to the priority queue
    for (int i = 0; i < numFiles; ++i) {
        std::string line;
        if (std::getline(inputFiles[i], line)) {
            std::istringstream iss(line);
            std::string term;
            int docID;
            int freq;
            iss >> term >> docID >> freq;
            pq.emplace(term, docID, freq, i);
        }
    }

    // Prepare output files
    std::ofstream outFile(outputIndexFile, std::ios::binary);
    if (!outFile.is_open()) {
        std::cerr << "Error: Unable to open output file for writing: " << outputIndexFile << std::endl;
        return;
    }
    std::ofstream lexiconOut(outputLexiconFile);
    if (!lexiconOut.is_open()) {
        std::cerr << "Error: Unable to open lexicon file for writing: " << outputLexiconFile << std::endl;
        return;
    }
    int64_t currentOffset = 0;

    // Variables to store postings for the current term
    std::string currentTerm = "";
    std::vector<int> docIDs;
    std::vector<int> freqs;

    // Block variables
    const int BLOCK_SIZE = 128; // Adjust as needed

    // Perform the multi-way merge
    while (!pq.empty()) {
        Posting topPosting = pq.top();
        pq.pop();

        // Check if we have moved to a new term
        if (currentTerm != topPosting.term) {
            // If not the first term, write the previous term's postings to disk
            if (!currentTerm.empty()) {
                // Write postings for the previous term
                // Split postings into blocks
                size_t numBlocks = (docIDs.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;
                size_t termStartOffset = outFile.tellp();

                // Write term size and term
                size_t termSize = currentTerm.size();
                outFile.write(reinterpret_cast<const char*>(&termSize), sizeof(size_t));
                outFile.write(currentTerm.c_str(), termSize);

                // Write number of blocks
                outFile.write(reinterpret_cast<const char*>(&numBlocks), sizeof(size_t));

                // For each block
                for (size_t blockIndex = 0; blockIndex < numBlocks; ++blockIndex) {
                    size_t start = blockIndex * BLOCK_SIZE;
                    size_t end = std::min(start + BLOCK_SIZE, docIDs.size());

                    // Extract block postings
                    std::vector<int> blockDocIDs(docIDs.begin() + start, docIDs.begin() + end);
                    std::vector<int> blockFreqs(freqs.begin() + start, freqs.begin() + end);

                    // Delta encode docIDs within block
                    std::vector<int> deltaDocIDs(blockDocIDs.size());
                    deltaDocIDs[0] = blockDocIDs[0];
                    for (size_t i = 1; i < blockDocIDs.size(); ++i) {
                        deltaDocIDs[i] = blockDocIDs[i] - blockDocIDs[i - 1];
                    }

                    // Compress docIDs and freqs separately
                    std::vector<std::uint8_t> encodedDocIDs;
                    for (int deltaDocID : deltaDocIDs) {
                        varByteEncode(deltaDocID, encodedDocIDs);
                    }

                    std::vector<std::uint8_t> encodedFreqs;
                    for (int freq : blockFreqs) {
                        varByteEncode(freq, encodedFreqs);
                    }

                    // Write sizes of docIDs and freqs blocks
                    size_t docIDsSize = encodedDocIDs.size();
                    size_t freqsSize = encodedFreqs.size();
                    outFile.write(reinterpret_cast<const char*>(&docIDsSize), sizeof(size_t));
                    outFile.write(reinterpret_cast<const char*>(&freqsSize), sizeof(size_t));

                    // Write the compressed blocks
                    outFile.write(reinterpret_cast<const char*>(encodedDocIDs.data()), docIDsSize);
                    outFile.write(reinterpret_cast<const char*>(encodedFreqs.data()), freqsSize);
                }

                // Update lexicon with term, offset, length, docFrequency
                int64_t newOffset = outFile.tellp();
                int32_t length = static_cast<int32_t>(newOffset - termStartOffset);
                int docFrequency = docIDs.size();

                lexiconOut << currentTerm << " " << termStartOffset << " " << length << " " << docFrequency << "\n";

                currentOffset = newOffset;
                docIDs.clear();
                freqs.clear();
            }

            // Reset variables for the new term
            currentTerm = topPosting.term;
        }

        // Add the current posting to the term's posting list
        docIDs.push_back(topPosting.docID);
        freqs.push_back(topPosting.freq);

        // Read the next posting from the same file and add it to the priority queue
        int fileIdx = topPosting.fileIndex;
        std::string line;
        if (std::getline(inputFiles[fileIdx], line)) {
            std::istringstream iss(line);
            std::string term;
            int docID;
            int freq;
            iss >> term >> docID >> freq;
            pq.emplace(term, docID, freq, fileIdx);
        }
    }

    // Write postings for the last term
    if (!currentTerm.empty()) {
        // Similar to above, write the last term's postings
        size_t numBlocks = (docIDs.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;
        size_t termStartOffset = outFile.tellp();

        size_t termSize = currentTerm.size();
        outFile.write(reinterpret_cast<const char*>(&termSize), sizeof(size_t));
        outFile.write(currentTerm.c_str(), termSize);

        // Write number of blocks
        outFile.write(reinterpret_cast<const char*>(&numBlocks), sizeof(size_t));

        // For each block
        for (size_t blockIndex = 0; blockIndex < numBlocks; ++blockIndex) {
            size_t start = blockIndex * BLOCK_SIZE;
            size_t end = std::min(start + BLOCK_SIZE, docIDs.size());

            // Extract block postings
            std::vector<int> blockDocIDs(docIDs.begin() + start, docIDs.begin() + end);
            std::vector<int> blockFreqs(freqs.begin() + start, freqs.begin() + end);

            // Delta encode docIDs within block
            std::vector<int> deltaDocIDs(blockDocIDs.size());
            deltaDocIDs[0] = blockDocIDs[0];
            for (size_t i = 1; i < blockDocIDs.size(); ++i) {
                deltaDocIDs[i] = blockDocIDs[i] - blockDocIDs[i - 1];
            }

            // Compress docIDs and freqs separately
            std::vector<std::uint8_t> encodedDocIDs;
            for (int deltaDocID : deltaDocIDs) {
                varByteEncode(deltaDocID, encodedDocIDs);
            }

            std::vector<std::uint8_t> encodedFreqs;
            for (int freq : blockFreqs) {
                varByteEncode(freq, encodedFreqs);
            }

            // Write sizes of docIDs and freqs blocks
            size_t docIDsSize = encodedDocIDs.size();
            size_t freqsSize = encodedFreqs.size();
            outFile.write(reinterpret_cast<const char*>(&docIDsSize), sizeof(size_t));
            outFile.write(reinterpret_cast<const char*>(&freqsSize), sizeof(size_t));

            // Write the compressed blocks
            outFile.write(reinterpret_cast<const char*>(encodedDocIDs.data()), docIDsSize);
            outFile.write(reinterpret_cast<const char*>(encodedFreqs.data()), freqsSize);
        }

        // Update lexicon with term, offset, length, docFrequency
        int64_t newOffset = outFile.tellp();
        int32_t length = static_cast<int32_t>(newOffset - termStartOffset);
        int docFrequency = docIDs.size();

        lexiconOut << currentTerm << " " << termStartOffset << " " << length << " " << docFrequency << "\n";
    }

    // Close all files
    for (int i = 0; i < numFiles; ++i) {
        inputFiles[i].close();
    }
    outFile.close();
    lexiconOut.close();
    std::cout << "[INFO] Merged inverted index and lexicon generated successfully." << std::endl;
}
