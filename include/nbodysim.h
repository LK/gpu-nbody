#include <math.h>
#include <stdlib.h>

#ifndef __NBODYSIM_H__
#define __NBODYSIM_H__

#ifndef __CUDACC__
#define __host__
#define __device__
#endif

typedef struct simdata_t {
  float *data; // position, velocity, features
  unsigned int posdim;
  unsigned int featdim;
  unsigned int nparticles;
} simdata_t;

__host__ __device__ inline unsigned int simdata_buf_size(simdata_t *sdata) {
  return (sdata->posdim * 2 + sdata->featdim) * sdata->nparticles;
}

__host__ __device__ inline simdata_t *simdata_create(unsigned int posdim,
                                                     unsigned int featdim,
                                                     unsigned int nparticles) {
  simdata_t *sdata = (simdata_t *)malloc(sizeof(simdata_t));
  sdata->posdim = posdim;
  sdata->featdim = featdim;
  sdata->nparticles = nparticles;
  sdata->data = (float *)malloc(sizeof(float) * simdata_buf_size(sdata));
  return sdata;
}

__host__ __device__ inline float *simdata_pos_ptr(simdata_t *sdata,
                                                  unsigned int idx) {
  return (sdata->data + idx * (sdata->posdim * 2 + sdata->featdim));
}

__host__ __device__ inline float *simdata_vel_ptr(simdata_t *sdata,
                                                  unsigned int idx) {
  return (sdata->data + idx * (sdata->posdim * 2 + sdata->featdim) +
          sdata->posdim);
}

__host__ __device__ inline float *simdata_feat_ptr(simdata_t *sdata,
                                                   unsigned int idx) {
  return (sdata->data + idx * (sdata->posdim * 2 + sdata->featdim) +
          2 * sdata->posdim);
}

__host__ __device__ inline void simdata_free(simdata_t *sdata) {
  free(sdata->data);
  free(sdata);
}

typedef enum integrator_t { INT_EULER = 0 } integrator_t;
typedef enum force_t { FORCE_NEWTONIAN = 0 } force_t;
typedef enum simulator_mode_t {
  MODE_SIMPLE = 0,
  MODE_CELESTIAL
} simulator_mode_t;

__host__ __device__ inline float get_multiplier(simulator_mode_t mode) {
  switch (mode) {
  case MODE_CELESTIAL:
    return (6.673 * pow(10, -11) * 13.3153474);
  default:
    return 1;
  }
}

void run_simulation(simdata_t *data, integrator_t int_type, force_t force_type,
                    simulator_mode_t mode, float time_step, int steps);

#endif
