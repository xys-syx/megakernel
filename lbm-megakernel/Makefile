# Standalone direct-Clang build. No Enzyme, MLIR, parent tree, or profiler needed.
CUDA_CXX ?= clang++
CUDA_PATH ?= /usr/local/cuda-12.9
SM ?= 120
PYTHON ?= python3
EXTRA_FLAGS ?=
FLAGS := -O3 -ffast-math -std=c++11 -fno-exceptions -DABI -DSIZE=20 \
         -Iinclude -I$(CUDA_PATH)/include --cuda-path=$(CUDA_PATH) \
         --cuda-gpu-arch=sm_$(SM) $(EXTRA_FLAGS)
LIBS := -L$(CUDA_PATH)/lib64 -lcudart_static -ldl -lrt -lpthread -lm
SOURCES := $(wildcard src/* kernels/* include/*)

.PHONY: all check bench report clean FORCE
all: build/lbm
build:
	mkdir -p $@
# Rebuild if the compiler/flags change, without rebuilding on every invocation.
build/config.txt: FORCE | build
	@{ $(CUDA_CXX) --version; printf '%s\n' '$(CUDA_CXX) $(FLAGS) $(LIBS)'; } > $@.tmp
	@cmp -s $@.tmp $@ && rm $@.tmp || mv $@.tmp $@
build/lbm: $(SOURCES) Makefile build/config.txt | build
	$(CUDA_CXX) $(FLAGS) src/main.cu -o $@ $(LIBS)
check: all
	$(PYTHON) scripts/validate.py
bench: all
	$(PYTHON) scripts/benchmark.py
report:
	$(PYTHON) scripts/report.py
clean:
	rm -rf build
