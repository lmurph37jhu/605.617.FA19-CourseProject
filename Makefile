SRC_DIR := src
OBJ_DIR := obj
BIN_DIR := bin

SRCS := $(shell find $(SRC_DIR) -name *.cu)
OBJS := $(SRCS:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)
EXEC := $(BIN_DIR)/doppler
NVCC_FLAGS := -L /usr/local/cuda/lib64 -lcudart -lcufft -std=c++11

$(EXEC): $(OBJS)
	mkdir -p $(dir $@)
	nvcc $(NVCC_FLAGS) -o $@ $(OBJS)
	@echo "Done"

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	mkdir -p $(dir $@)
	nvcc $(NVCC_FLAGS) -o $@ -c $< -I$(SRC_DIR)
