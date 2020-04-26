CC = gcc


all: nbodysim

nbodysim: nbodysim.c
	$(CC) -o $@ $^



