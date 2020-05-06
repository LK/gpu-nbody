#include <stdio.h>
#include <stdlib.h>

#define NUM_IMAGES 1797
#define IMAGE_DIM 64
#define PERPLEXITY 50
#define MIN_SD 0.001
#define MAX_SD 1000

void load_images();
void load_labels();
void load_mnist();
void showDigit(int index);
int *getLabels();
float **getImages();
