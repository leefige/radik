ifeq ($(OS), Windows_NT)
    SUDO	:=
else ifeq ($(USER),root)
    SUDO	:=
else
    SUDO	:= sudo
endif

ifeq ($(OS), Windows_NT)
    CWD		:= $(shell echo %cd%)
    USER	:= $(shell echo %username%)
else
    CWD		:= $(shell pwd)
    USER	:= $(shell whoami)
endif

# NOTE: change this to your GPU compute capability version
# e.g.:
# 	T4, Geforce RTX 2080	: 75
# 	A100, A30				: 80
# 	A10, GeForce RTX 3090	: 86
# 	L40, GeForce RTX 4090	: 89
# 	H100					: 90
SM_VERSION	?=	75

# NOTE: you may want to change these variables
PLOT_DIR	?= 	$(CWD)/plot

# NOTE: you may want to select which GPU to use
GPU_FLAGS	?=	--gpus '"device=0"'

RUN_FLAGS	:=	--name $(USER)-radik --rm $(GPU_FLAGS)
MNT_FLAGS	:=  --mount type=bind,src=$(PLOT_DIR),dst=/radik/plot

.PHONY: build
build:
	$(SUDO) docker build --build-arg SM_VERSION=$(SM_VERSION) -t radik .

.PHONY: run
run:
	$(SUDO) docker run -it $(RUN_FLAGS) $(MNT_FLAGS) radik

.PHONY: exec
exec:
	$(SUDO) docker run $(RUN_FLAGS) $(MNT_FLAGS) radik $(DOCKER_CMD)
