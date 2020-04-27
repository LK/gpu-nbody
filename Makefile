CC = gcc
CFLAGS = "-Iinclude"
SRC = src
BIN = bin

BASE_SRCS := src/simulator/simdata.c

all: test-cpu

test-cpu: $(BASE_SRCS) src/simulator/cpu/nbodysim.c src/test/test.c
	$(CC) -o bin/$@ $(CFLAGS) $^

clean:
	rm -r $(BIN)/*
