
NVCC = nvcc
TARGET = fnn

CUDA_SRC = main.cu $(wildcard src/*.cu)
CPP_SRC = $(wildcard src/*.cpp)

OBJ = $(CUDA_SRC:.cu=.o) $(CPP_SRC:.cpp=.o)

NVCCFLAGS = -O2 -std=c++17 -Iinclude

MODE ?= optimized
BLOCK ?= 16

all: $(TARGET)

$(TARGET): $(OBJ)
	$(NVCC) $(NVCCFLAGS) $(OBJ) -o $(TARGET)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.o: %.cpp
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET) $(MODE) $(BLOCK)

clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all run clean
