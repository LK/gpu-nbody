#include "mnist.h"
#include "nbodysim.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

__device__ float getl2NormDiff(float *a, float *b, int dim) {
  float norm = 0;
  for (int i = 0; i < dim; i++) {
    float diff = a[i] - b[i];
    norm += diff * diff;
  }
  return norm;
}

__device__ float getPji(int dataSize, int i, int j, float sd,
                        float *l2NormDiffs) {
  float denominator = 0;
  for (int k = 0; k < dataSize; k++) {
    if (k == i)
      continue;
    denominator += exp(-l2NormDiffs[i * dataSize + k] / 2 / sd / sd);
  }
  float numerator = exp(-l2NormDiffs[i * dataSize + j] / 2 / sd / sd);
  return numerator / denominator;
}

__device__ float getPerp(int dataSize, int i, float sd, float *l2NormDiffs) {
  float shEntr = 0;
  for (int j = 0; j < dataSize; j++) {
    float pji = getPji(dataSize, i, j, sd, l2NormDiffs);
    shEntr -= (pji * log2(pji));
  }
  return pow(2.0, shEntr);
}

__device__ float findSd(int dataSize, int i, float *l2NormDiffs) {
  float minSd = MIN_SD;
  float maxSD = MAX_SD;
  float currentPerp = getPerp(dataSize, i, (minSd + maxSD) / 2, l2NormDiffs);
  while (fabsf(PERPLEXITY - currentPerp) > 0.01) {
    if (currentPerp > PERPLEXITY) {
      maxSD = (minSd + maxSD) / 2;
    } else {
      minSd = (minSd + maxSD) / 2;
    }
    currentPerp = getPerp(dataSize, i, (minSd + maxSD) / 2, l2NormDiffs);
  }
  float sd = (maxSD + minSd) / 2;
  return sd;
}

__global__ void _tsne_precompute_aux_kernel(simdata_t *d_sdata, float *aux,
                                            float *l2NormDiffs) {
  int idx = threadIdx.x + blockIdx.x * 1024;
  if (idx >= d_sdata->nparticles)
    return;
  float stdev = findSd(d_sdata->nparticles, idx, l2NormDiffs);
  float *myAux = aux + d_sdata->nparticles * idx;
  for (int i = 0; i < d_sdata->nparticles; i++) {
    myAux[i] = getPji(d_sdata->nparticles, i, idx, stdev, l2NormDiffs);
  }
}

__global__ void _tsne_precompute_l2norms_kernel(simdata_t *d_sdata,
                                                float *l2NormDiffs) {
  int idx = threadIdx.x + blockIdx.x * 1024;
  if (idx >= d_sdata->nparticles)
    return;
  float *myFeatures = simdata_feat_ptr(d_sdata, idx);
  float *myL2NormDiffs = l2NormDiffs + d_sdata->nparticles * idx;
  for (int i = 0; i < d_sdata->nparticles; i++) {
    myL2NormDiffs[i] = getl2NormDiff(simdata_feat_ptr(d_sdata, i), myFeatures,
                                     d_sdata->featdim);
  }
}

__host__ float *tsne_precompute(simdata_t *d_sdata, int nparticles) {

  float *l2NormDiffs;
  cudaMalloc(&l2NormDiffs, sizeof(float) * (nparticles * nparticles));
  _tsne_precompute_l2norms_kernel<<<nparticles / 1024 + 1, 1024>>>(d_sdata,
                                                                   l2NormDiffs);

  float *aux;
  cudaMalloc(&aux, sizeof(float) * (nparticles * nparticles + 1));
  _tsne_precompute_aux_kernel<<<nparticles / 1024 + 1, 1024>>>(d_sdata, aux,
                                                               l2NormDiffs);

  cudaFree(l2NormDiffs);
  return aux;
}

__global__ void _tsne_precompute_denominators_kernel(simdata_t *d_sdata,
                                                     float *denominators) {
  int idx = threadIdx.x + blockIdx.x * 1024;
  if (idx >= d_sdata->nparticles)
    return;
  float denominator = 0;
  for (int i = 0; i < d_sdata->nparticles; i++) {
    if (i != idx) {
      denominator += 1 / (1 + getl2NormDiff(simdata_pos_ptr(d_sdata, i),
                                            simdata_pos_ptr(d_sdata, idx),
                                            d_sdata->posdim));
    }
  }
  denominators[idx] = denominator;
}

__global__ void _tsne_sum_denominators_kernel(float *denominators, float *aux,
                                              int nparticles) {
  float denominator = 0;
  for (int i = 0; i < nparticles; i++) {
    denominator += denominators[i];
  }
  aux[nparticles * nparticles] = denominator;
}

__host__ void setQjiDenominator(simdata_t *d_sdata, float *aux,
                                int nparticles) {
  float *denominators;
  cudaMalloc(&denominators, sizeof(float) * nparticles);
  _tsne_precompute_denominators_kernel<<<nparticles / 1024 + 1, 1024>>>(
      d_sdata, denominators);
  _tsne_sum_denominators_kernel<<<1, 1>>>(denominators, aux, nparticles);
  cudaFree(denominators);
}

__device__ void tsne_compute(int particleA, int particleB, float *d_force,
                             float *d_position, float *d_positionActor,
                             simdata_t *d_sdata, float *aux) {
  int dataSize = d_sdata->nparticles;
  float pji = aux[particleA * dataSize + particleB];
  float pij = aux[particleB * dataSize + particleA];
  float qjiNumerator =
      1 / (1 + getl2NormDiff(d_position, d_positionActor, d_sdata->posdim));
  float qjiDenominator = aux[d_sdata->nparticles * d_sdata->nparticles];

  float scalar =
      4 * ((pij + pji) / 2 / dataSize - qjiNumerator / qjiDenominator) /
      (1 + getl2NormDiff(d_position, d_positionActor, d_sdata->posdim));

  for (int i = 0; i < d_sdata->posdim; i++) {
    d_force[i] += scalar * (d_positionActor[i] - d_position[i]);
  }
}
