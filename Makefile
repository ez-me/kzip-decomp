# Reconstructed KZIP command-line compressor.
#
# Build:
#   make            build build/kzip
#   make clean      remove build artifacts
#
# Requires a C++20 compiler. There are no external library dependencies.
#
# The reconstruction targets the behavior of the statically linked
# `kzip-static` binary (SHA-256
# e7bd2055a45df7cd2785d564474916be07205364985924bc2808d6048c0063cd).
# Documentation for the individual components lives in the source comments.

CXX ?= c++
CXXFLAGS ?= -std=c++20 -O2 -Wall -Wextra -Wpedantic
CPPFLAGS ?= -Isrc
BUILD_DIR ?= build

# Emit header dependencies so a changed struct layout cannot leave a stale
# object behind.
DEPFLAGS := -MMD -MP

SOURCES := \
	src/kzip_cli.cpp \
	src/strategy0_encoder.cpp \
	src/strategy12_parser.cpp \
	src/strategy0_model.cpp \
	src/parser_cost.cpp \
	src/huffman.cpp \
	src/deflate_writer.cpp \
	src/block_splitter.cpp \
	src/split_score_model.cpp \
	src/zip_writer.cpp

OBJECTS := $(patsubst src/%.cpp,$(BUILD_DIR)/%.o,$(SOURCES))
DEPS := $(OBJECTS:.o=.d)

.PHONY: all clean

all: $(BUILD_DIR)/kzip

$(BUILD_DIR)/kzip: $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(BUILD_DIR)/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(DEPFLAGS) -c -o $@ $<

clean:
	rm -rf $(BUILD_DIR)

-include $(DEPS)
