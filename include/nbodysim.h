#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

#ifndef __NBODYSIM_H__
#define __NBODYSIM_H__

#ifndef __CUDACC__
#define __host__
#define __device__
#endif

/// Store data for simulation.
typedef struct simdata_t {
  /// A pointer to an array of bodies. Each body stores its position, its
  /// velocity, and its features, in that order (as floats).
  float *data;

  /// The number of dimensions used for the particle position and velocity.
  unsigned int posdim;

  /// The number of feature dimensions that each particle has.
  unsigned int featdim;

  /// The number of particles in our system.
  unsigned int nparticles;
} simdata_t;

/// Compute the size of a `simdata_t` data buffer.
__host__ __device__ inline unsigned int simdata_buf_size(simdata_t *sdata) {
  return (sdata->posdim * 2 + sdata->featdim) * sdata->nparticles;
}

/// Initialize a `simdata_t` on the heap, with an allocated but uninitialized
/// data array.
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

/// Get a pointer to the position of a particle.
__host__ __device__ inline float *simdata_pos_ptr(simdata_t *sdata,
                                                  unsigned int idx) {
  return (sdata->data + idx * (sdata->posdim * 2 + sdata->featdim));
}

/// Get a pointer to the velocity of a particle.
__host__ __device__ inline float *simdata_vel_ptr(simdata_t *sdata,
                                                  unsigned int idx) {
  return (sdata->data + idx * (sdata->posdim * 2 + sdata->featdim) +
          sdata->posdim);
}

/// Get a pointer to the feature vector of a particle.
__host__ __device__ inline float *simdata_feat_ptr(simdata_t *sdata,
                                                   unsigned int idx) {
  return (sdata->data + idx * (sdata->posdim * 2 + sdata->featdim) +
          2 * sdata->posdim);
}

/// Clean up a `simdata_t`.
__host__ __device__ inline void simdata_free(simdata_t *sdata) {
  free(sdata->data);
  free(sdata);
}

typedef enum integrator_t { INT_EULER = 0, INT_LEAPFROG = 1 } integrator_t;
typedef enum force_t {
  /// Traditional Newtonian force calculation.
  FORCE_NEWTONIAN = 0,

  /// Newtonian force calculation with G = 1.
  FORCE_NEWTONIAN_SIMPLE = 1
} force_t;

/// Defines additional configuration options for the simulation.
typedef struct simconfig_t {
  /// If true, try to precompute force information before executing simulation.
  bool precompute;
} simconfig_t;

void run_simulation(simdata_t *sdata, simconfig_t *sconfig,
                    integrator_t int_type, force_t force_type, float dt,
                    int steps);

#endif
