CC = gcc
CFLAGS = -Iinclude -O3 -std=gnu99 -lm -g

NVCC = nvcc
NVCCFLAGS = -Iinclude -O3 -lm
GENCODE = -gencode=arch=compute_37,code=\"sm_37,compute_37\"

BASE_SRCS := src/test/timing.c
TEST_SIMPLE_SRCS := $(BASE_SRCS) src/test/test-simple.c
TEST_RANDOM_SRCS := $(BASE_SRCS) src/test/test-random.c
TEST_CELESTIAL_SRCS := $(BASE_SRCS) src/test/test-celestial.c src/simulator/solarsystemdata.c
TEST_GALAXY_SRCS := $(BASE_SRCS) src/test/test-galaxy.c
TEST_TSNE_SRCS := $(BASE_SRCS) src/test/test-sne.c

CPU_EXECUTABLES := test-simple-cpu test-random-cpu test-celestial-cpu test-galaxy-cpu test-tsne-cpu
GPU_EXECUTABLES := test-simple-gpu test-random-gpu test-celestial-gpu test-galaxy-gpu

CPU_SRCS := src/simulator/cpu/nbodysim.c
GPU_SRCS := src/simulator/gpu/nbodysim.o

all: $(CPU_EXECUTABLES) $(GPU_EXECUTABLES)

cpu: $(CPU_EXECUTABLES)

gpu: $(GPU_EXECUTABLES)

test-simple-cpu: $(TEST_SIMPLE_SRCS) $(CPU_SRCS)
	$(CC) -o bin/$@ $(CFLAGS) $^

test-random-cpu: $(TEST_RANDOM_SRCS) $(CPU_SRCS)
	$(CC) -o bin/$@ $(CFLAGS) $^

test-celestial-cpu: $(TEST_CELESTIAL_SRCS) $(CPU_SRCS)
	$(CC) -o bin/$@ $(CFLAGS) $^

test-galaxy-cpu: $(TEST_GALAXY_SRCS) $(CPU_SRCS)
	$(CC) -o bin/$@ $(CFLAGS) $^

test-tsne-cpu: $(TEST_TSNE_SRCS) $(CPU_SRCS)
	$(CC) -o bin/$@ $(CFLAGS) $^

test-simple-gpu: $(patsubst %.c,%.o,$(TEST_SIMPLE_SRCS)) $(GPU_SRCS)
	$(NVCC) $(GENCODE) -o bin/$@ $(NVCCFLAGS) $^

test-random-gpu: $(patsubst %.c,%.o,$(TEST_RANDOM_SRCS)) $(GPU_SRCS)
	$(NVCC) $(GENCODE) -o bin/$@ $(NVCCFLAGS) $^

test-celestial-gpu: $(patsubst %.c,%.o,$(TEST_CELESTIAL_SRCS)) $(GPU_SRCS)
	$(NVCC) $(GENCODE) -o bin/$@ $(NVCCFLAGS) $^

test-galaxy-gpu: $(patsubst %.c,%.o,$(TEST_GALAXY_SRCS)) $(GPU_SRCS)
	$(NVCC) $(GENCODE) -o bin/$@ $(NVCCFLAGS) $^

%.o: %.c
	$(NVCC) $(GENCODE) -x cu $(NVCCFLAGS) -dc $< -o $@

%.o: %.cu
	$(NVCC) $(GENCODE) -x cu $(NVCCFLAGS) -dc $< -o $@

clean:
	find . -name "*.o" -type f -delete
	find bin/ -not -name '\.*' -type f -delete
