CXX = g++
CXXFLAGS = -std=c++14 -Wall -O3 -march=native -fPIC
INCLUDES = -I../../

TARGET = test_hnsw
SRC = test.cc

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< -pthread

clean:
	rm -f $(TARGET) *.o *.bin

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
