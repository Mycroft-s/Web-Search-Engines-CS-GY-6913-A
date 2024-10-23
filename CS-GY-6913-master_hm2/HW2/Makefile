# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++11

# Executable names
PARSER = parser
MERGER = merger
QUERY_PROCESSOR = query_processor

# Source files for each executable
PARSER_SOURCES = parser_main.cpp parser.cpp
MERGER_SOURCES = merger_main.cpp merger.cpp
QUERY_PROCESSOR_SOURCES = query_main.cpp query.cpp index_api.cpp

# Default
all: $(PARSER) $(MERGER) $(QUERY_PROCESSOR)

# build parser
$(PARSER): $(PARSER_SOURCES)
	$(CXX) $(CXXFLAGS) -o $(PARSER) $(PARSER_SOURCES)

# build merger
$(MERGER): $(MERGER_SOURCES)
	$(CXX) $(CXXFLAGS) -o $(MERGER) $(MERGER_SOURCES)

# build query_processor
$(QUERY_PROCESSOR): $(QUERY_PROCESSOR_SOURCES)
	$(CXX) $(CXXFLAGS) -o $(QUERY_PROCESSOR) $(QUERY_PROCESSOR_SOURCES)

# Clean
clean:
	rm -f $(PARSER) $(MERGER) $(QUERY_PROCESSOR)

# Phony targets
.PHONY: all clean
