#include "nbodysim.h"
#include "forces.cu"
#include "integrators.cu"
#include "timing.h"
#include "tsneforces.cu"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

/// Clone a data buffer from the CPU to the GPU.
simdata_t *simdata_clone_cpu_gpu(simdata_t *sdata) {
  simdata_t *d_sdata;
  float *d_sdata_data;
  cudaMalloc(&d_sdata, sizeof(simdata_t));
  cudaMalloc(&d_sdata_data, sizeof(float) * simdata_buf_size(sdata));
  float *tmp = sdata->data;
  sdata->data = d_sdata_data;
  cudaMemcpy(d_sdata, sdata, sizeof(simdata_t), cudaMemcpyHostToDevice);
  sdata->data = tmp;
  cudaMemcpy(d_sdata_data, sdata->data, sizeof(float) * simdata_buf_size(sdata),
             cudaMemcpyHostToDevice);
  return d_sdata;
}

/// Replace a CPU data buffer with data from the GPU.
void simdata_copy_gpu_cpu(simdata_t *d_sdata, simdata_t *sdata) {
  float *data = sdata->data;
  cudaMemcpy(sdata, d_sdata, sizeof(simdata_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(data, sdata->data, sizeof(float) * simdata_buf_size(sdata),
             cudaMemcpyDeviceToHost);
  sdata->data = data;
}

/// Free data buffer from GPU.
void simdata_gpu_free(simdata_t *d_sdata) {
  simdata_t copy;
  cudaMemcpy(&copy, d_sdata, sizeof(simdata_t), cudaMemcpyDeviceToHost);
  cudaFree(d_sdata);
  cudaFree(copy.data);
}

/// Main force computation GPU kernel.
__global__ void compute_acceleration(simdata_t *d_sdata, float *d_accel,
                                     force_t force_type, float *aux) {
  int particleIdx = threadIdx.x + blockIdx.x * 1024;
  if (particleIdx >= d_sdata->nparticles)
    return;
  float *d_position = simdata_pos_ptr(d_sdata, particleIdx);
  float *d_features = simdata_feat_ptr(d_sdata, particleIdx);

  for (int j = 0; j < d_sdata->nparticles; j++) {
    float *d_positionActor = simdata_pos_ptr(d_sdata, j);
    float *d_featuresActor = simdata_feat_ptr(d_sdata, j);

    if (particleIdx != j) {
      if (force_type == FORCE_TSNE) {
        tsne_compute(particleIdx, j, d_accel + particleIdx * d_sdata->posdim,
                     d_position, d_positionActor, d_sdata, aux);
      } else {
        newtonian_compute(
            particleIdx, j, d_accel + particleIdx * d_sdata->posdim, d_position,
            d_features, d_positionActor, d_featuresActor, d_sdata, aux);
      }
    }
  }

  for (int j = 0; j < d_sdata->posdim; j++) {
    switch (force_type) {
    case FORCE_TSNE:
      break;
    case FORCE_NEWTONIAN:
      d_accel[particleIdx * d_sdata->posdim + j] *=
          (6.673 * pow(10, -11) * 13.3153474) / d_features[0];
      break;
    case FORCE_NEWTONIAN_SIMPLE:
      d_accel[particleIdx * d_sdata->posdim + j] /= d_features[0];
      break;
    }
  }
}

/// Entrypoint to simulation on GPU.
__host__ void run_simulation(simdata_t *sdata, simconfig_t *sconfig,
                             integrator_t int_type, force_t force_type,
                             float time_step, int steps) {
  simdata_t *d_sdata = simdata_clone_cpu_gpu(sdata);
  float *d_accel;
  cudaMalloc(&d_accel, sizeof(float) * sdata->posdim * sdata->nparticles);
  cudaMemset(d_accel, 0, sizeof(float) * sdata->posdim * sdata->nparticles);

  double integration_time = 0;
  double force_calc_time = 0;
  measure_t *fulltimer = start_timer();
  measure_t *timer = start_timer();
  float *aux = NULL;
  if (sconfig->precompute) {
    switch (force_type) {
    case FORCE_TSNE:
      aux = tsne_precompute(d_sdata, sdata->nparticles);
      break;
    case FORCE_NEWTONIAN:
    case FORCE_NEWTONIAN_SIMPLE:
      aux = newtonian_precompute(d_sdata, sdata->nparticles);
      break;
    }
  }
  double precompute_time = end_timer_silent(timer);

  for (int step = 0; step < steps; step++) {
    cudaThreadSynchronize();
    timer = start_timer();
    if (int_type == INT_LEAPFROG) {
      leapfrog_integrate<<<sdata->nparticles / 1024 + 1, 1024>>>(
          d_sdata, d_accel, time_step, true);
    }

    if (force_type == FORCE_TSNE) {
      setQjiDenominator(d_sdata, aux, sdata->nparticles);
    }

    cudaThreadSynchronize();
    integration_time += end_timer_silent(timer);

    if (step > 0) {
      cudaMemset(d_accel, 0, sizeof(float) * sdata->posdim * sdata->nparticles);
    }

    cudaThreadSynchronize();
    timer = start_timer();
    compute_acceleration<<<sdata->nparticles / 1024 + 1, 1024>>>(
        d_sdata, d_accel, force_type, aux);
    cudaThreadSynchronize();
    force_calc_time += end_timer_silent(timer);

    timer = start_timer();
    cudaThreadSynchronize();
    switch (int_type) {
    case INT_EULER:
      euler_integrate<<<sdata->nparticles / 1024 + 1, 1024>>>(d_sdata, d_accel,
                                                              time_step);
      break;
    case INT_LEAPFROG:
      leapfrog_integrate<<<sdata->nparticles / 1024 + 1, 1024>>>(
          d_sdata, d_accel, time_step, false);
      break;
    }
    cudaThreadSynchronize();
    integration_time += end_timer_silent(timer);
  }
  double fulltime = end_timer_silent(fulltimer);
  printf("%f, %f, %f, %f\n", precompute_time, force_calc_time, integration_time,
         fulltime);

  simdata_copy_gpu_cpu(d_sdata, sdata);
  simdata_gpu_free(d_sdata);
  cudaFree(d_accel);
  if (aux)
    cudaFree(aux);
}
