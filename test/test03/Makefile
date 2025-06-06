CXX = g++
CXXFLAGS = -std=c++14 -O3 -Wall -Wextra -march=native
INCLUDES = -I../../
LIBS = -lpthread

all: build_index search

build_index: build_index.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LIBS)

search: search.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LIBS)

clean:
	rm -f build_index search

.PHONY: all clean 