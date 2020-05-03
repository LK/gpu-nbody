#include "nbodysim.h"
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

__global__ void leapfrog_integrate(simdata_t *d_sdata, float *d_acceleration,
                                   float dt, bool before_accel_update) {
  float *d_pos = simdata_pos_ptr(d_sdata, threadIdx.x);
  float *d_vel = simdata_vel_ptr(d_sdata, threadIdx.x);
  float *d_accel = d_acceleration + d_sdata->posdim * threadIdx.x;

  for (int i = 0; i < d_sdata->posdim; i++) {
    d_vel[i] += d_accel[i] * dt * 0.5;
    if (before_accel_update) {
      d_pos[i] += d_vel[i] * dt;
    }
  }
}

__global__ void euler_integrate(simdata_t *d_sdata, float *d_acceleration,
                                float dt) {
  float *d_pos = simdata_pos_ptr(d_sdata, threadIdx.x);
  float *d_vel = simdata_vel_ptr(d_sdata, threadIdx.x);
  float *d_accel = d_acceleration + d_sdata->posdim * threadIdx.x;

  for (int i = 0; i < d_sdata->posdim; i++) {
    d_pos[i] += d_vel[i] * dt + 0.5 * d_accel[i] * dt * dt;
    d_vel[i] += d_accel[i] * dt;
  }
}
