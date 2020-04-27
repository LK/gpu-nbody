#include "nbodysim.h"
#include "stdlib.h"

simdata_t *simdata_create(unsigned int posdim, unsigned int featdim,
                          unsigned int nparticles) {
  simdata_t *sdata = (simdata_t *)malloc(sizeof(simdata_t));
  sdata->posdim = posdim;
  sdata->featdim = featdim;
  sdata->nparticles = nparticles;
  sdata->data =
      (float *)malloc(sizeof(float) * (posdim * 2 + featdim) * nparticles);
  return sdata;
}

void simdata_free(simdata_t *sdata) {
  free(sdata->data);
  free(sdata);
}
