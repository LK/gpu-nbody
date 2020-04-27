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

inline float *simdata_pos_ptr(simdata_t *sdata, unsigned int idx) {
  return (sdata->data + idx * (sdata->posdim * 2 + sdata->featdim));
}

inline float *simdata_vel_ptr(simdata_t *sdata, unsigned int idx) {
  return (sdata->data + idx * (sdata->posdim * 2 + sdata->featdim) +
          sdata->posdim);
}

inline float *simdata_feat_ptr(simdata_t *sdata, unsigned int idx) {
  return (sdata->data + idx * (sdata->posdim * 2 + sdata->featdim) +
          2 * sdata->posdim);
}

void simdata_free(simdata_t *sdata) {
  free(sdata->data);
  free(sdata);
}
