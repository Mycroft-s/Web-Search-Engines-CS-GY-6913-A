#include <string>

void startQueryProcessor(const std::string& indexFilePath, const std::string& lexiconFilePath, const std::string& queryFilePath, const std::string& outputFilePath);

int main() {
    std::string indexFilePath = "tmp/final_inverted_index.bin";
    std::string lexiconFilePath = "tmp/lexicon.txt";
    std::string queryFilePath = "../queries/queries.eval.one.small.tsv";
    std::string outputFilePath = "bm25_results.txt"; // Output results file

    startQueryProcessor(indexFilePath, lexiconFilePath, queryFilePath, outputFilePath);
    return 0;
}
