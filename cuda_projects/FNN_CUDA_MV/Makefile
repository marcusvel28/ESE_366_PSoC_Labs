NVCC = nvcc
TARGET = fnn

SRC = main.cu $(wildcard src/*.cu)
OBJ = $(SRC:.cu=.o)

NVCCFLAGS = -O2 -std=c++17 -Iinclude

all: $(TARGET)

$(TARGET): $(OBJ)
	$(NVCC) $(NVCCFLAGS) $(OBJ) -o $(TARGET)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all run clean