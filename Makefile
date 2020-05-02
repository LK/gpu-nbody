CC = gcc
CFLAGS = -Iinclude -O3 -std=gnu99 -lm -g

NVCC = nvcc
NVCCFLAGS = -Iinclude -O3 -lm
GENCODE = -gencode=arch=compute_37,code=\"sm_37,compute_37\"

TEST_SIMPLE_SRCS := src/test/test-simple.c
TEST_CELESTIAL_SRCS := src/test/test-celestial.c src/simulator/solarsystemdata.c
TEST_GALAXY_SRCS := src/test/test-galaxy.c

all: test-simple-cpu test-simple-gpu test-celestial-cpu test-celestial-gpu test-galaxy-cpu

test-simple-cpu: $(TEST_SIMPLE_SRCS) src/simulator/cpu/nbodysim.c
	$(CC) -o bin/$@ $(CFLAGS) $^

test-celestial-cpu: $(TEST_CELESTIAL_SRCS) src/simulator/cpu/nbodysim.c
	$(CC) -o bin/$@ $(CFLAGS) $^

test-galaxy-cpu: $(TEST_GALAXY_SRCS) src/simulator/cpu/nbodysim.c
	$(CC) -o bin/$@ $(CFLAGS) $^

test-simple-gpu: $(patsubst %.c,%.o,$(TEST_SIMPLE_SRCS)) src/simulator/gpu/nbodysim.o
	$(NVCC) $(GENCODE) -o bin/$@ $(NVCCFLAGS) $^

test-celestial-gpu: $(patsubst %.c,%.o,$(TEST_CELESTIAL_SRCS)) src/simulator/gpu/nbodysim.o
	$(NVCC) $(GENCODE) -o bin/$@ $(NVCCFLAGS) $^

%.o: %.c
	$(NVCC) $(GENCODE) -x cu $(NVCCFLAGS) -dc $< -o $@

%.o: %.cu
	$(NVCC) $(GENCODE) -x cu $(NVCCFLAGS) -dc $< -o $@

clean:
	find . -name "*.o" -type f -delete
	find bin/ -not -name '\.*' -type f -delete
