CUDA_SM_VERSION ?= 75

CUDA_PATH       ?= /usr/local/cuda
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin

NVCC            ?= $(CUDA_BIN_PATH)/nvcc
AR              ?= ar

CXXFLAGS 	:= -std=c++14 -O3
CUDAFLAGS 	:= -arch=sm_$(CUDA_SM_VERSION) -Xptxas=-v

SRC_DIR 	:= src
TEST_DIR 	:= test
OBJ_DIR 	:= obj
LIB_DIR		:= lib

LDFLAGS   	:= -lcurand
INCLUDES 	:= -I$(SRC_DIR) -I$(TEST_DIR)

OBJS 	:=  $(OBJ_DIR)/blockselect/BlockSelectFloat32.o		\
			$(OBJ_DIR)/blockselect/BlockSelectFloat64.o		\
			$(OBJ_DIR)/blockselect/BlockSelectFloat128.o	\
			$(OBJ_DIR)/blockselect/BlockSelectFloat256.o	\
			$(OBJ_DIR)/blockselect/BlockSelectFloatF512.o	\
			$(OBJ_DIR)/blockselect/BlockSelectFloatF1024.o	\
			$(OBJ_DIR)/blockselect/BlockSelectFloatF2048.o	\
			$(OBJ_DIR)/blockselect/BlockSelectFloatT512.o	\
			$(OBJ_DIR)/blockselect/BlockSelectFloatT1024.o	\
			$(OBJ_DIR)/blockselect/BlockSelectFloatT2048.o	\
			$(OBJ_DIR)/BlockSelectFloat.o

LIBS 	:= libblockselect.a

TARGETS := test_block_select.out

.PHONY: all
all: $(TARGETS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(@D)
	$(NVCC) $(INCLUDES) $(CXXFLAGS) $(CUDAFLAGS) -c $< -o $@

%.a: $(OBJS)
	@mkdir -p $(LIB_DIR)
	@rm -rf $(LIB_DIR)/$@
	$(foreach var, $(OBJS), $(AR) cr $@ $(var);)
	@mv $@ $(LIB_DIR)/$@

test_%.out: $(TEST_DIR)/test_%.cu $(LIBS)
	$(NVCC) $(INCLUDES) $(CXXFLAGS) $(CUDAFLAGS) $(LDFLAGS) $< -o $@ -L$(LIB_DIR) -lblockselect

.PHONY: clean
clean:
	rm -rf $(OBJ_DIR)

.PHONY: cleandist
cleandist:
	rm -rf $(TARGETS) $(LIB_DIR)

.PHONY: cleanall
cleanall: clean cleandist
