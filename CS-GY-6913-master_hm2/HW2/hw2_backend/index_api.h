#ifndef INDEX_API_H
#define INDEX_API_H

#include <string>
#include <unordered_map>
#include <fstream>
#include <vector>
#include <cstdint>

// Structure to hold lexicon entries
struct LexiconEntry {
    int64_t offset;
    int32_t length;
    int docFrequency;
};

// Forward declaration
class InvertedList;

// IndexAPI class definition
class IndexAPI {
public:
    std::unordered_map<std::string, LexiconEntry> lexicon;

    IndexAPI(const std::string& indexFilePath, const std::string& lexiconFilePath);
    ~IndexAPI();

    InvertedList* openList(const std::string& term);
    void closeList(InvertedList* invList);

private:
    std::string indexFilePath; // Store index file path

    void loadLexicon(const std::string& lexiconFilePath);
};

class InvertedList {
public:
    InvertedList(const std::string& term, const std::string& indexFilePath, const LexiconEntry& lexEntry);
    ~InvertedList();

    // Primitives
    bool hasNext();
    int nextGEQ(int targetDocID); // Returns next docID >= targetDocID or INT32_MAX
    double getScore();            // Returns the term frequency of the current posting

private:
    std::ifstream indexFile;      // Each InvertedList has its own file stream
    LexiconEntry lexEntry;
    size_t numBlocks;
    size_t currentBlockIndex;
    size_t postingIndexInBlock;
    bool endOfList;

    std::vector<int> docIDs;
    std::vector<int> freqs;
    int currentDocID;
    int currentFreq;

    size_t bytesRead;             // Tracks the number of bytes read
    size_t totalBytes;            // Total bytes to read for this inverted list

    void loadNextBlock();
    int varByteDecode(std::istream& in);
};

#endif // INDEX_API_H
