CC = gcc
CFLAGS = -Iinclude -O3 -std=c99 -lm

NVCC = nvcc
NVCCFLAGS = -Iinclude -O3 -lm
GENCODE = -gencode=arch=compute_37,code=\"sm_37,compute_37\"

all: test-cpu test-gpu

test-cpu: src/simulator/solarsystemdata.c src/simulator/cpu/nbodysim.c src/test/test.c
	$(CC) -o bin/$@ $(CFLAGS) $^

test-gpu: src/simulator/solarsystemdata.o src/simulator/gpu/nbodysim.o src/test/test.o
	$(NVCC) $(GENCODE) -o bin/$@ $(NVCCFLAGS) $^

%.o: %.c
	$(NVCC) $(GENCODE) -x cu $(NVCCFLAGS) -dc $< -o $@

%.o: %.cu
	$(NVCC) $(GENCODE) -x cu $(NVCCFLAGS) -dc $< -o $@

clean:
	find . -name "*.o" -type f -delete
	find bin/ -not -name '\.*' -type f -delete
