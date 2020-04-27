CC = gcc
CFLAGS = -Iinclude -O3 -std=c99 -lm

NVCC = nvcc
NVCCFLAGS = -Iinclude -O3 -lm
GENCODE = -gencode=arch=compute_37,code=\"sm_37,compute_37\"

BASE_SRCS := src/simulator/simdata.c

all: test-cpu

test-cpu: $(BASE_SRCS) src/simulator/cpu/nbodysim.c src/test/test.c
	$(CC) -o bin/$@ $(CFLAGS) $^

test-gpu: $(BASE_SRCS) src/simulator/gpu/nbodysim.c src/test/test.c
	$(NVCC) $(GENCODE) -o bin/$@ $(NVCCFLAGS) $^

clean:
	rm -r bin/*
