#include "index_api.h"
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cstdint>

// Variable-byte decoding function writen by mycroft
int InvertedList::varByteDecode(std::istream& in) {
    int number = 0;
    int shift = 0;
    while (true) {
        int byte = in.get();
        if (byte == EOF) {
            // Handle EOF
            return -1;
        }
        if (byte & 0x80) {
            number |= (byte & 0x7F) << shift;
            break;
        } else {
            number |= byte << shift;
            shift += 7;
        }
    }
    return number;
}

// IndexAPI implementation
IndexAPI::IndexAPI(const std::string& indexFilePath, const std::string& lexiconFilePath)
    : indexFilePath(indexFilePath) {
    std::ifstream testIndexFile(indexFilePath, std::ios::binary);
    if (!testIndexFile.is_open()) {
        std::cerr << "Error: Unable to open index file: " << indexFilePath << std::endl;
        return;
    }
    testIndexFile.close();
    loadLexicon(lexiconFilePath);
}

IndexAPI::~IndexAPI() {
    // Destructor
}

void IndexAPI::loadLexicon(const std::string& lexiconFilePath) {
    std::ifstream inFile(lexiconFilePath);
    if (!inFile.is_open()) {
        std::cerr << "Error opening lexicon file!" << std::endl;
        return;
    }

    std::string term;
    int64_t offset;
    int32_t length;
    int docFrequency;

    while (inFile >> term >> offset >> length >> docFrequency) {
        lexicon[term] = { offset, length, docFrequency };
    }

    inFile.close();
}

InvertedList* IndexAPI::openList(const std::string& term) {
    auto it = lexicon.find(term);
    if (it == lexicon.end()) {
        // Term not found
        return nullptr;
    }
    return new InvertedList(term, indexFilePath, it->second);
}

void IndexAPI::closeList(InvertedList* invList) {
    delete invList;
}

// InvertedList implementation
InvertedList::InvertedList(const std::string& term, const std::string& indexFilePath, const LexiconEntry& lexEntry)
    : indexFile(indexFilePath, std::ios::binary), lexEntry(lexEntry), currentBlockIndex(0),
      postingIndexInBlock(0), endOfList(false), bytesRead(0) {

    if (!indexFile.is_open()) {
        std::cerr << "Error: Unable to open index file: " << indexFilePath << std::endl;
        endOfList = true;
        return;
    }

    // Load term metadata from the index file
    indexFile.seekg(lexEntry.offset, std::ios::beg);

    // Read term size and term
    size_t termSize;
    if (!indexFile.read(reinterpret_cast<char*>(&termSize), sizeof(size_t))) {
        std::cerr << "Error reading term size for term: " << term << std::endl;
        endOfList = true;
        return;
    }
    bytesRead += sizeof(size_t);

    std::string storedTerm(termSize, '\0');
    if (!indexFile.read(&storedTerm[0], termSize)) {
        std::cerr << "Error reading term string for term: " << term << std::endl;
        endOfList = true;
        return;
    }
    bytesRead += termSize;

    if (storedTerm != term) {
        std::cerr << "Term mismatch at offset " << lexEntry.offset << ": expected '" << term << "', found '" << storedTerm << "'" << std::endl;
        endOfList = true;
        return;
    }

    // Read number of blocks
    if (!indexFile.read(reinterpret_cast<char*>(&numBlocks), sizeof(size_t))) {
        std::cerr << "Error reading number of blocks for term: " << term << std::endl;
        endOfList = true;
        return;
    }
    bytesRead += sizeof(size_t);

    // Calculate totalBytes to read based on lexEntry.length
    totalBytes = lexEntry.length;

    // Start by loading the first block
    loadNextBlock();
}

InvertedList::~InvertedList() {
    if (indexFile.is_open()) {
        indexFile.close();
    }
}

bool InvertedList::hasNext() {
    return !endOfList;
}

int InvertedList::nextGEQ(int targetDocID) {
    while (hasNext()) {
        if (postingIndexInBlock >= docIDs.size()) {
            loadNextBlock();
            if (endOfList) {
                return INT32_MAX; // Representing MAXDID
            }
        }

        if (postingIndexInBlock >= docIDs.size()) {
            // No more postings in the current block
            continue;
        }

        // Binary search within the current block for efficiency
        auto it = std::lower_bound(docIDs.begin() + postingIndexInBlock, docIDs.end(), targetDocID);
        if (it != docIDs.end()) {
            size_t index = std::distance(docIDs.begin(), it);
            currentDocID = *it;
            currentFreq = freqs[index];
            postingIndexInBlock = index + 1;
            return currentDocID;
        } else {
            loadNextBlock();
        }
    }
    return INT32_MAX; // Representing MAXDID
}

double InvertedList::getScore() {
    // Returns term frequency as a double
    return static_cast<double>(currentFreq);
}

void InvertedList::loadNextBlock() {
    // Load the next block from the index file
    if (currentBlockIndex >= numBlocks) {
        endOfList = true;
        return;
    }

    // Read sizes of docIDs and freqs blocks
    size_t docIDsSize;
    size_t freqsSize;
    if (!indexFile.read(reinterpret_cast<char*>(&docIDsSize), sizeof(size_t))) {
        std::cerr << "Error reading docIDsSize from index file." << std::endl;
        endOfList = true;
        return;
    }
    bytesRead += sizeof(size_t);

    if (!indexFile.read(reinterpret_cast<char*>(&freqsSize), sizeof(size_t))) {
        std::cerr << "Error reading freqsSize from index file." << std::endl;
        endOfList = true;
        return;
    }
    bytesRead += sizeof(size_t);

    // Sanity checks for block sizes
    const size_t MAX_BLOCK_SIZE = 100 * 1024 * 1024; // 100 MB
    if (docIDsSize > MAX_BLOCK_SIZE || freqsSize > MAX_BLOCK_SIZE) {
        std::cerr << "Error: Block size too large. docIDsSize: " << docIDsSize << ", freqsSize: " << freqsSize << std::endl;
        endOfList = true;
        return;
    }

    // Check if reading these sizes would exceed totalBytes
    if (bytesRead + docIDsSize + freqsSize > totalBytes) {
        std::cerr << "Error: Attempting to read beyond the specified length in lexicon. bytesRead: " << bytesRead
                  << ", docIDsSize + freqsSize: " << (docIDsSize + freqsSize)
                  << ", totalBytes: " << totalBytes << std::endl;
        endOfList = true;
        return;
    }

    // Read compressed data
    std::vector<std::uint8_t> compressedDocIDs(docIDsSize);
    std::vector<std::uint8_t> compressedFreqs(freqsSize);
    if (!indexFile.read(reinterpret_cast<char*>(compressedDocIDs.data()), docIDsSize)) {
        std::cerr << "Error reading compressedDocIDs from index file." << std::endl;
        endOfList = true;
        return;
    }
    bytesRead += docIDsSize;

    if (!indexFile.read(reinterpret_cast<char*>(compressedFreqs.data()), freqsSize)) {
        std::cerr << "Error reading compressedFreqs from index file." << std::endl;
        endOfList = true;
        return;
    }
    bytesRead += freqsSize;

    // Decompress docIDs
    docIDs.clear();
    std::string docIDData(reinterpret_cast<char*>(compressedDocIDs.data()), compressedDocIDs.size());
    std::istringstream docIDStream(docIDData);
    int docID = 0;
    while (docIDStream.tellg() < static_cast<std::streampos>(docIDData.size())) {
        int deltaDocID = varByteDecode(docIDStream);
        if (deltaDocID == -1) {
            std::cerr << "Error decoding deltaDocID in docIDs." << std::endl;
            break;
        }
        docID += deltaDocID;
        docIDs.push_back(docID);
    }

    // Decompress freqs
    freqs.clear();
    std::string freqData(reinterpret_cast<char*>(compressedFreqs.data()), compressedFreqs.size());
    std::istringstream freqStream(freqData);
    while (freqStream.tellg() < static_cast<std::streampos>(freqData.size())) {
        int freq = varByteDecode(freqStream);
        if (freq == -1) {
            std::cerr << "Error decoding frequency in freqs." << std::endl;
            break;
        }
        freqs.push_back(freq);
    }

    // Ensure that docIDs and freqs have the same size
    if (docIDs.size() != freqs.size()) {
        std::cerr << "Error: Mismatch between docIDs and freqs sizes. docIDs: " << docIDs.size()
                  << ", freqs: " << freqs.size() << std::endl;
        endOfList = true;
        return;
    }

    postingIndexInBlock = 0;
    currentBlockIndex++;

    // Debug: Print the number of postings loaded
    std::cout << "Loaded block " << currentBlockIndex << "/" << numBlocks << " with " << docIDs.size() << " postings." << std::endl;
}
