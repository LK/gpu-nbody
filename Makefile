CC = gcc
CFLAGS = "-Iinclude"
SRC = src
BIN = bin

all: test-cpu

test-cpu: src/simulator/cpu/nbodysim.c src/test/test.c
	$(CC) -o bin/$@ $(CFLAGS) $^

clean:
	rm -r $(BIN)/*
