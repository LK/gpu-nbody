CC = gcc
CFLAGS = -Iinclude -O3 -std=gnu99 -lm -g

NVCC = nvcc
NVCCFLAGS = -Iinclude -O3 -lm
GENCODE = -gencode=arch=compute_37,code=\"sm_37,compute_37\"

TEST_SIMPLE_SRCS := src/test/test-simple.c
TEST_RANDOM_SRCS := src/test/test-random.c
TEST_CELESTIAL_SRCS := src/test/test-celestial.c src/simulator/solarsystemdata.c
TEST_GALAXY_SRCS := src/test/test-galaxy.c

all: test-simple-cpu test-simple-gpu test-random-cpu test-random-gpu test-celestial-cpu test-celestial-gpu test-galaxy-cpu test-galaxy-gpu

cpu: test-simple-cpu test-random-cpu test-celestial-cpu test-galaxy-cpu

gpu: test-simple-gpu test-random-gpu test-celestial-gpu test-galaxy-gpu

test-simple-cpu: $(TEST_SIMPLE_SRCS) src/simulator/cpu/nbodysim.c
	$(CC) -o bin/$@ $(CFLAGS) $^

test-random-cpu: $(TEST_RANDOM_SRCS) src/simulator/cpu/nbodysim.c
	$(CC) -o bin/$@ $(CFLAGS) $^

test-celestial-cpu: $(TEST_CELESTIAL_SRCS) src/simulator/cpu/nbodysim.c
	$(CC) -o bin/$@ $(CFLAGS) $^

test-galaxy-cpu: $(TEST_GALAXY_SRCS) src/simulator/cpu/nbodysim.c
	$(CC) -o bin/$@ $(CFLAGS) $^

test-simple-gpu: $(patsubst %.c,%.o,$(TEST_SIMPLE_SRCS)) src/simulator/gpu/nbodysim.o
	$(NVCC) $(GENCODE) -o bin/$@ $(NVCCFLAGS) $^

test-random-gpu: $(patsubst %.c,%.o,$(TEST_RANDOM_SRCS)) src/simulator/gpu/nbodysim.o
	$(NVCC) $(GENCODE) -o bin/$@ $(NVCCFLAGS) $^

test-celestial-gpu: $(patsubst %.c,%.o,$(TEST_CELESTIAL_SRCS)) src/simulator/gpu/nbodysim.o
	$(NVCC) $(GENCODE) -o bin/$@ $(NVCCFLAGS) $^

test-galaxy-gpu: $(patsubst %.c,%.o,$(TEST_GALAXY_SRCS)) src/simulator/gpu/nbodysim.o
	$(NVCC) $(GENCODE) -o bin/$@ $(NVCCFLAGS) $^

%.o: %.c
	$(NVCC) $(GENCODE) -x cu $(NVCCFLAGS) -dc $< -o $@

%.o: %.cu
	$(NVCC) $(GENCODE) -x cu $(NVCCFLAGS) -dc $< -o $@

clean:
	find . -name "*.o" -type f -delete
	find bin/ -not -name '\.*' -type f -delete
