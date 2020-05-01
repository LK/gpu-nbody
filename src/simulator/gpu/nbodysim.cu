#include "nbodysim.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__device__ void compute_force(float *d_force, float *d_position,
                              float *d_features, float *d_positionActor,
                              float *d_featuresActor, simdata_t *d_sdata) {

  float feature = d_features[0];
  float featureActor = d_featuresActor[0];

  float distance = 0;
  for (int i = 0; i < d_sdata->posdim; i++) {
    float pos = d_position[i];
    float posActor = d_positionActor[i];
    float deltaPos = pos - posActor;
    distance += deltaPos * deltaPos;
  }
  distance = sqrt(distance);

  for (int i = 0; i < d_sdata->posdim; i++) {
    float pos = d_position[i];
    float posActor = d_positionActor[i];
    float deltaPos = pos - posActor;

    float f =
        feature * featureActor / distance / distance * deltaPos / distance;
    d_force[i] -= f;
  }
}

__global__ void integrator(simdata_t *d_sdata, float *d_acceleration,
                           float time_step) {
  float *d_pos = simdata_pos_ptr(d_sdata, threadIdx.x);
  float *d_vel = simdata_vel_ptr(d_sdata, threadIdx.x);
  float *d_accel = d_acceleration + d_sdata->posdim * threadIdx.x;
  for (int i = 0; i < d_sdata->posdim; i++) {
    d_pos[i] += d_vel[i] * time_step + 0.5 * d_accel[i] * time_step * time_step;
    d_vel[i] += d_accel[i] * time_step;
  }
}

__host__ void dump(simdata_t *sdata, int step) {
  printf("=============== STEP %d ===============\n", step);
  float *pos, *vel;
  for (int i = 0; i < sdata->nparticles; i++) {
    pos = simdata_pos_ptr(sdata, i);
    vel = simdata_vel_ptr(sdata, i);
    printf("\t(%.04f, %.04f)\t(%.04f, %.04f)\n", pos[0], pos[1], vel[0],
           vel[1]);
  }
}

simdata_t *simdata_clone_cpu_gpu(simdata_t *sdata) {
  simdata_t *d_sdata;
  float *d_sdata_data;
  cudaMalloc(&d_sdata, sizeof(simdata_t));
  cudaMalloc(&d_sdata_data, sizeof(float) * simdata_buf_size(sdata));
  float *tmp = sdata->data;
  sdata->data = d_sdata_data;
  cudaMemcpy(d_sdata, sdata, sizeof(simdata_t), cudaMemcpyHostToDevice);
  sdata->data = tmp;
  cudaMemcpy(d_sdata_data, sdata->data,
             sizeof(float) * simdata_buf_size(sdata), cudaMemcpyHostToDevice);
  return d_sdata;
}

void simdata_copy_gpu_cpu(simdata_t *d_sdata, simdata_t *sdata) {
  float *data = sdata->data;
  cudaMemcpy(sdata, d_sdata, sizeof(simdata_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(data, sdata->data, sizeof(float) * simdata_buf_size(sdata),
      cudaMemcpyDeviceToHost);
  sdata->data = data;
}

void simdata_gpu_free(simdata_t *d_sdata) {
  simdata_t copy;
  cudaMemcpy(&copy, d_sdata, sizeof(simdata_t), cudaMemcpyDeviceToHost);
  cudaFree(d_sdata);
  cudaFree(copy.data);
}

__global__ void compute_acceleration(simdata_t *d_sdata, float *d_accel,
                                     force_t force_type, float multiplier) {
  int particleIdx = threadIdx.x;
  float *d_position = simdata_pos_ptr(d_sdata, particleIdx);
  float *d_features = simdata_feat_ptr(d_sdata, particleIdx);

  for (int j = 0; j < d_sdata->nparticles; j++) {
    float *d_positionActor = simdata_pos_ptr(d_sdata, j);
    float *d_featuresActor = simdata_feat_ptr(d_sdata, j);

    if (particleIdx != j) {
      compute_force(d_accel + particleIdx * d_sdata->posdim, d_position,
                    d_features, d_positionActor, d_featuresActor, d_sdata);
    }
  }

  for (int j = 0; j < d_sdata->posdim; j++) {
    d_accel[particleIdx * d_sdata->posdim + j] *= multiplier / d_features[0];
  }
}

__host__ void run_simulation(simdata_t *sdata, integrator_t int_type,
                             force_t force_type, simulator_mode_t mode,
                             float time_step, int steps) {
  simdata_t *d_sdata = simdata_clone_cpu_gpu(sdata);
  float *d_accel;
  cudaMalloc(&d_accel, sizeof(float) * sdata->posdim * sdata->nparticles);
  for (int step = 0; step < steps; step++) {
    cudaMemset(&d_accel, 0, sizeof(float) * sdata->posdim * sdata->nparticles);

    compute_acceleration<<<1, sdata->nparticles>>>(d_sdata, d_accel,
                                                   force_type,
                                                   get_multiplier(mode));

    integrator<<<1, sdata->nparticles>>>(d_sdata, d_accel, time_step);
  }

  simdata_copy_gpu_cpu(d_sdata, sdata);
  simdata_gpu_free(d_sdata);
  cudaFree(d_accel);
}
