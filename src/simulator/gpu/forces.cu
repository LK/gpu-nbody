#include "nbodysim.h"
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

__global__ void _newtonian_precompute_kernel(simdata_t *d_sdata, float *aux) {
  float myMass = simdata_feat_ptr(d_sdata, threadIdx.x)[0];
  float *myAux = aux + d_sdata->nparticles * threadIdx.x;
  for (int i = 0; i < d_sdata->nparticles; i++) {
    myAux[i] = myMass * simdata_feat_ptr(d_sdata, i)[0];
  }
}

__host__ float *newtonian_precompute(simdata_t *d_sdata, int nparticles) {
  float *aux;
  cudaMalloc(&aux, sizeof(float) * nparticles * nparticles);
  _newtonian_precompute_kernel<<<1, nparticles>>>(d_sdata, aux);
  return aux;
}

__device__ void newtonian_compute(
    int particleA, int particleB, float *d_force, float *d_position,
    float *d_features, float *d_positionActor, float *d_featuresActor,
    simdata_t *d_sdata, float *aux) {
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

    float massTerm = aux ? aux[particleA * d_sdata->nparticles + particleB] :
      d_features[0] * d_featuresActor[0];

    float f = massTerm / distance / distance * deltaPos / distance;
    d_force[i] -= f;
  }
}
