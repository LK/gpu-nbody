#include "nbodysim.h"
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define PERPLEXITY 50
#define MIN_SD 0.00000001
#define MAX_SD 100000000

__device__ float getl2NormDiff(float *a, float *b, int dim)
{
    float norm = 0;
    for (int i = 0; i < dim; i++)
    {
        float diff = a[i] - b[i];
        norm += diff * diff;
    }
    return norm;
}
__device__ float getPji(int dataSize, int i, int j, float sd, float *l2NormDiffs)
{
    float denominator = 0;
    for (int k = 0; k < dataSize; k++)
    {
        if (k == i)
            continue;
        denominator += exp(-l2NormDiffs[i * dataSize + k] / 2 / sd / sd);
    }
    float numerator = exp(-l2NormDiffs[i * dataSize + j] / 2 / sd / sd);
    return numerator / denominator;
}
__device__ float getPerp(int dataSize, int i, float sd, float *l2NormDiffs)
{
    float shEntr = 0;
    for (int j = 0; j < dataSize; j++)
    {
        float pji = getPji(dataSize, i, j, sd, l2NormDiffs);
        shEntr -= (pji * log2(pji));
    }
    return pow(2.0, shEntr);
}
__device__ float findSd(int dataSize, int i, float *l2NormDiffs)
{
    float minSd = MIN_SD;
    float maxSD = MAX_SD;
    float currentPerp = getPerp(dataSize, i, (minSd + maxSD) / 2, l2NormDiffs);
    while (fabsf(PERPLEXITY - currentPerp) > 0.01)
    {
        if (currentPerp > PERPLEXITY)
        {
            maxSD = (minSd + maxSD) / 2;
        }
        else
        {
            minSd = (minSd + maxSD) / 2;
        }
        currentPerp = getPerp(dataSize, i, (minSd + maxSD) / 2, l2NormDiffs);
    }
    float sd = (maxSD + minSd) / 2;
    return sd;
}
__global__ void _tsne_precompute_aux_kernel(simdata_t *d_sdata, float *aux, float *l2NormDiffs) {
    float stdev = findSd(d_sdata->nparticles, threadIdx.x, l2NormDiffs);
    float *myAux = aux + d_sdata->nparticles * threadIdx.x;
    for (int i = 0; i < d_sdata->nparticles; i++)
    {
        myAux[i] = getPji(d_sdata->nparticles, i, threadIdx.x, stdev, l2NormDiffs);
    }
}
__global__ void _tsne_precompute_l2norms_kernel(simdata_t *d_sdata, float *l2NormDiffs) {
    float *myFeatures = simdata_feat_ptr(d_sdata, threadIdx.x);
    float *myL2NormDiffs = l2NormDiffs + d_sdata->nparticles * threadIdx.x;
    for(int i = 0; i < d_sdata->nparticles; i++)
    {
        myL2NormDiffs[i] = getl2NormDiff(simdata_feat_ptr(d_sdata, i), myFeatures, d_sdata->featdim);
    }
}

__host__ float *tsne_precompute(simdata_t *d_sdata, int nparticles) {

  float *l2NormDiffs;
  cudaMalloc(&l2NormDiffs, sizeof(float) * (nparticles * nparticles));
  _tsne_precompute_l2norms_kernel<<<1, nparticles>>>(d_sdata, l2NormDiffs);

  float *aux;
  cudaMalloc(&aux, sizeof(float) * (nparticles * nparticles + 1));
  _tsne_precompute_aux_kernel<<<1, nparticles>>>(d_sdata, aux, l2NormDiffs);

  cudaFree(l2NormDiffs);
  return aux;
}


// TODO: Parallelize
__host__ float getl2NormDiffQiDen(float *a, float *b, int dim)
{
    float norm = 0;
    for (int i = 0; i < dim; i++)
    {
        float diff = a[i] - b[i];
        norm += diff * diff;
    }
    return norm;
}
__host__ void setQjiDenominator(simdata_t *d_sdata, float *aux)
{
    // float denominator = 0;
    // for (int k = 0; k < d_sdata->nparticles; k++)
    // {
    //     for (int l = 0; l < d_sdata->nparticles; l++)
    //     {
    //         if (l != k)
    //             denominator += 1 / (1 + getl2NormDiffQiDen(simdata_pos_ptr(d_sdata, l), simdata_pos_ptr(d_sdata, k), d_sdata->posdim));
    //     }
    // }
    // printf("%f\n", denominator);
    // aux[(d_sdata->nparticles) * (d_sdata->nparticles)] = denominator;
}

__device__ void tsne_compute(int particleA, int particleB, float *d_force, float *d_position, float *d_positionActor, simdata_t *d_sdata, float *aux) {

    int dataSize = d_sdata->nparticles;
    float pji = aux[particleA * dataSize + particleB];
    float pij = aux[particleB * dataSize + particleA];
    float qjiNumerator = 1 / (1 + getl2NormDiff(d_position, d_positionActor, d_sdata->posdim));
    // float qjiDenominator = aux[d_sdata->nparticles * d_sdata->nparticles];
    float qjiDenominator = 17000;

    float scalar = 4 * ((pij + pji) / 2 / dataSize - qjiNumerator / qjiDenominator) / (1 + getl2NormDiff(d_position, d_positionActor, d_sdata->posdim));
    
    for (int i = 0; i < d_sdata->posdim; i++) {  
        d_force[i] += scalar * (d_positionActor[i] - d_position[i]);
    }

}