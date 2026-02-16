CXX ?= g++
CXXFLAGS ?= -O3 -march=native -std=c++20 -Wall -Wextra -Wpedantic
EIGEN_DIR ?= ../third_party/eigen

BIN := eigen_transform_chain_bench
SRC := eigen_transform_chain_bench.cpp

.PHONY: all clean run

all: $(BIN)

$(BIN): $(SRC)
	$(CXX) $(CXXFLAGS) -I $(EIGEN_DIR) $< -o $@

run: $(BIN)
	./$(BIN)

clean:
	rm -f $(BIN)
