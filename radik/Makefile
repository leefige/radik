CUDA_SM_VERSION ?= 75

CUDA_PATH       ?= /usr/local/cuda
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin

NVCC            ?= $(CUDA_BIN_PATH)/nvcc

CXXFLAGS 	:= -std=c++14 -O3
CUDAFLAGS 	:= -arch=sm_$(CUDA_SM_VERSION)
LDFLAGS   	:= -lcurand

TEST_DIR 	:= test
OBJ_DIR 	:= obj

RADIK_DEPS	:= $(wildcard RadixSelect/*)
PQ_DEPS		:= $(wildcard PQ/*)
ABLATION_DEPS	:= 	$(wildcard RadixSelect/*)
ABLATION_DEPS	+= 	$(wildcard ablation/*)

TARGETS 	:= test_radik.out test_PQ.out test_ablation.out

.PHONY: all
all: $(TARGETS)

test_radik.out: $(TEST_DIR)/test_radik.cu $(RADIK_DEPS)
	$(NVCC) $(CXXFLAGS) $(CUDAFLAGS) $(LDFLAGS) $< -o $@ -Xptxas=-v

test_PQ.out: $(TEST_DIR)/test_PQ.cu $(PQ_DEPS)
	$(NVCC) $(CXXFLAGS) $(CUDAFLAGS) $(LDFLAGS) $< -o $@ -Xptxas=-v

test_ablation.out: $(TEST_DIR)/test_ablation.cu $(ABLATION_DEPS)
	$(NVCC) $(CXXFLAGS) $(CUDAFLAGS) $(LDFLAGS) $< -o $@ -Xptxas=-v

.PHONY: clean
clean:
	rm -rf $(TARGETS)
