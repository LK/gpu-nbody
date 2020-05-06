#include "mnist.h"
#include "nbodysim.h"
#include "timing.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int *LABELS;
float **IMAGES;

void dumpt(simdata_t *sdata, int step) {
  printf("=============== STEP %d ===============\n", step);
  float *pos, *vel;
  for (int i = 0; i < sdata->nparticles; i++) {
    pos = simdata_pos_ptr(sdata, i);
    vel = simdata_vel_ptr(sdata, i);
    printf("\t(%.04f, %.04f)\t(%.04f, %.04f)\n", pos[0], pos[1], vel[0],
           vel[1]);
  }
}

void dumpEmbedding(simdata_t *sdata) {
  FILE *fp;
  fp = fopen("output", "w");
  float *pos;
  for (int index = 0; index < sdata->nparticles; index++) {
    pos = simdata_pos_ptr(sdata, index);
    for (int i = 0; i < sdata->posdim; i++) {
      fprintf(fp, "%1.1f, ", pos[i]);
    }
    fprintf(fp, "%d\n", LABELS[index]);
  }
  fclose(fp);
}

int main() {
  load_mnist();
  IMAGES = getImages();
  LABELS = getLabels();

  int dataSize = 1024;     // number of data entries
  int dataDim = IMAGE_DIM; // size of mnist data entry
  int dim = 2;             // dimensional reduction size

  simdata_t *sdata = simdata_create(dim, dataDim, dataSize);

  for (int i = 0; i < dataSize; i++) {
    float *currentBuffer =
        &(sdata->data[i * (2 * sdata->posdim + sdata->featdim)]);
    for (int j = 0; j < dim; j++) {
      currentBuffer[j] = (float)rand() / (float)(RAND_MAX / 2) - 1;
      currentBuffer[dim + j] = 0;
    }
    for (int j = 0; j < dataDim; j++)
      currentBuffer[dim * 2 + j] = IMAGES[i][j];
  }

  simconfig_t sconfig = {.precompute = true};
  measure_t *timer = start_timer();
  run_simulation(sdata, &sconfig, INT_EULER, FORCE_TSNE, 1, 2500);
  double runtime = end_timer(timer);
  dumpEmbedding(sdata);
  simdata_free(sdata);

  return 0;
}
