
#include "mnist.h"

int* LABEL_SET;
float** IMAGE_SET;

void load_images()
{
    FILE* file = fopen("data/mnist.dat", "r"); 
    IMAGE_SET = (float **)malloc(sizeof(float *) * NUM_IMAGES);
    char line[256];
    for (int i = 0; i < NUM_IMAGES; i++)
    {
        float * curr = (float *)malloc(sizeof(float) * IMAGE_DIM);
        for (int j = 0; j < 64; j++)
        {
            fgets(line, sizeof(line), file);
            curr[j] = atof(line);
        }
        IMAGE_SET[i] = curr;
    }
    fclose(file);

}
void load_labels()
{
    FILE* file = fopen("data/mnist_labels.dat", "r"); 
    LABEL_SET = (int *)malloc(sizeof(int) * NUM_IMAGES);
    char line[256];
    for(int i = 0; i < NUM_IMAGES; i++)
    {
        fgets(line, sizeof(line), file);
        LABEL_SET[i] = atoi(line);
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
        float val = IMAGE_SET[index][i];
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

    printf("Label: %d\n", LABEL_SET[index]);
}

int* getLabels()
{
    return LABEL_SET;
}

float** getImages()
{
    return IMAGE_SET;
}