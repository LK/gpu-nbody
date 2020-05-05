#include <stdio.h>
#include <stdlib.h>

#define NUM_IMAGES 1797

int LABELS[NUM_IMAGES];
float FLAT_IMAGES[NUM_IMAGES * 64];
float IMAGES[NUM_IMAGES][64];

void load_images()
{
    FILE* file = fopen("data/mnist.dat", "r"); 
    char line[256];
    for (int i = 0; i < NUM_IMAGES; i++)
    {
        for (int j = 0; j < 64; j++)
        {
            fgets(line, sizeof(line), file);
            IMAGES[i][j] = atof(line);
        }
    }
    fclose(file);

}
void load_labels()
{
    FILE* file = fopen("data/mnist_labels.dat", "r"); 
    char line[256];
    for(int i = 0; i < NUM_IMAGES; i++)
    {
        fgets(line, sizeof(line), file);
        LABELS[i] = atoi(line);
    }
    fclose(file);
}
void load_mnist()
{
    load_images();
    load_labels();
}

void showDigit(int index)
{
    int i;
    for (i = 0; i < 64; i++)
    {
        float val = IMAGES[index][i];
        if (val)
        {
            printf("%1.1f ", val);
        }
        else
        {
            printf("    ");
        }

        if ((i + 1) % 8 == 0)
            putchar('\n');
    }

    printf("Label: %d\n", LABELS[index]);
}