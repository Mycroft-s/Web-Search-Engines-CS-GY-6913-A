#include <string>

void startQueryProcessor(const std::string& indexFilePath, const std::string& lexiconFilePath);

int main() {
    std::string indexFilePath = "tmp/final_inverted_index.bin";
    std::string lexiconFilePath = "tmp/lexicon.txt";
    startQueryProcessor(indexFilePath, lexiconFilePath);
    return 0;
}
