#include <string>

void parseDocuments(const std::string& inputFilePath, const std::string& tempFilePrefix);

int main() {
    std::string inputFilePath = "collection.tsv";
    std::string tempFilePrefix = "tmp/temp_postings_";
    parseDocuments(inputFilePath, tempFilePrefix);
    return 0;
}
